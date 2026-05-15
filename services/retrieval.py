"""Catalog-backed retrieval engine with semantic (FAISS) and keyword search.

Combines sentence-transformer embeddings with deterministic keyword scoring,
role-aware re-ranking, query expansion, language-fit penalties, and duplicate
suppression to produce a ranked list of SHL catalog assessments.  Falls back
to pure keyword search when FAISS or the embedding model is unavailable.
"""

import json
import os
import pickle
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.helpers import (
    catalog_item_text,
    clean_text,
    format_recommendation,
    infer_requested_type_codes,
    item_has_any_code,
    normalize_for_match,
    parse_test_type_codes,
    tokenize,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG_PATH = ROOT_DIR / "catalog" / "shl_catalog.json"
DEFAULT_INDEX_PATH = ROOT_DIR / "embeddings" / "faiss.index"
DEFAULT_METADATA_PATH = ROOT_DIR / "embeddings" / "metadata.pkl"


TECH_ROLE_TERMS = {
    "developer", "engineer", "software", "backend", "back end", "frontend",
    "front end", "fullstack", "full stack", "programmer", "devops", "qa",
    "tester", "testing", "technical", "coding", "programming", "api",
    "rest", "restful", "cloud", "platform", "site reliability", "sre",
}


TECH_STACK_TERMS = {
    "java": 26.0,
    "python": 26.0,
    "sql": 20.0,
    "spring boot": 24.0,
    "spring": 20.0,
    "django": 18.0,
    "rest api": 20.0,
    "restful": 18.0,
    "api": 14.0,
    "javascript": 22.0,
    "c#": 22.0,
    ".net": 20.0,
    "ado.net": 18.0,
    "asp": 16.0,
    "aws": 15.0,
    "amazon web services": 15.0,
    "docker": 14.0,
    "kubernetes": 18.0,
    "devops": 14.0,
    "linux": 14.0,
    "sap": 10.0,
    "agile": 8.0,
    "manual testing": 10.0,
    "testing": 8.0,
}


SOFTWARE_ITEM_TERMS = {
    "java", "python", "sql", "javascript", "c#", ".net", "asp", "software",
    "programming", "database", "agile", "testing", "aws", "amazon web services",
    "linux", "sap", "query", "framework", "developer", "spring", "hibernate",
    "rest", "restful", "api", "web services", "kubernetes", "cluster",
    "cloud", "devops", "deployment", "monitoring", "scalability",
}


BACKEND_ITEM_TERMS = {
    "java", "python", "sql", "database", "aws", ".net", "c#", "asp", "api",
    "server", "rest", "restful", "web services", "spring", "django",
    "kubernetes", "cloud",
}


COGNITIVE_QUERY_TERMS = {
    "cognitive", "ability", "aptitude", "reasoning", "numerical", "verbal",
    "deductive", "inductive", "logical", "problem solving", "general ability",
}


PERSONALITY_QUERY_TERMS = {
    "personality", "opq", "behavior", "behaviour", "behavioral",
    "behavioural", "culture", "motivation",
}


COMMUNICATION_QUERY_TERMS = {
    "communication", "communicate", "interpersonal", "soft skills",
    "collaboration", "teamwork",
}


SALES_SERVICE_ITEM_TERMS = {
    "sales", "service", "customer", "contact center", "phone", "call",
}


# Query aliases expand modern engineering terms into SHL catalog language. They do
# not create products; they only improve matching against real catalog items.
QUERY_EXPANSIONS = {
    "java": "core java java 8 java frameworks spring hibernate jdbc oop concurrency backend programming concepts algorithms",
    "spring boot": "java frameworks spring hibernate backend rest restful web services api",
    "spring": "java frameworks spring hibernate backend rest api",
    "python": "python programming databases modules library django backend programming concepts",
    "django": "python programming backend rest api web framework databases",
    "sql": "sql database queries data manipulation transaction processing relational database",
    "rest api": "restful web services rest features api architecture requests responses security interceptors backend",
    "rest": "restful web services api architecture requests responses security backend",
    "api": "restful web services api architecture requests responses backend",
    "aws": "amazon web services aws cloud delivery monitoring logging security scalability devops",
    "amazon web services": "aws cloud delivery monitoring logging security scalability devops",
    "docker": "containers container orchestration kubernetes devops cloud deployment scalability",
    "kubernetes": "kubernetes architecture cluster services containers orchestration devops",
    "devops": "aws kubernetes cloud deployment monitoring logging scalability security",
    "backend": "backend developer server api database sql java python programming concepts logical reasoning",
    "backend developer": "backend developer server api database sql java python programming concepts logical reasoning",
    "software engineer": "software engineer programming concepts algorithms data structures java python sql logical reasoning problem solving",
}


STRATEGIC_TECH_PRODUCTS = {
    "programming concepts",
    "core java advanced level new",
    "java 8 new",
    "java frameworks new",
    "java design patterns new",
    "python new",
    "sql new",
    "restful web services new",
    "amazon web services aws development new",
    "kubernetes new",
    "shl verify interactive g+",
    "shl verify interactive deductive reasoning",
    "shl verify interactive inductive reasoning",
    "verify numerical ability",
    "global skills assessment",
}


CORE_PERSONALITY_PRODUCTS = {
    "occupational personality questionnaire opq32r",
    "opq universal competency report 2.0",
}


class RetrievalEngine:
    """Catalog-backed retrieval with FAISS when available and keyword fallback."""

    def __init__(
        self,
        catalog_path: Path = DEFAULT_CATALOG_PATH,
        index_path: Path = DEFAULT_INDEX_PATH,
        metadata_path: Path = DEFAULT_METADATA_PATH,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.model_name = model_name
        self.catalog = self._load_catalog()
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self.model = None
        self._load_index()

    def _load_catalog(self) -> List[Dict[str, Any]]:
        if not self.catalog_path.exists():
            return []
        try:
            with self.catalog_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict) and item.get("name") and item.get("url")]
        except Exception:
            return []
        return []

    def _load_index(self) -> None:
        if not self.index_path.exists() or not self.metadata_path.exists():
            return
        if os.path.getsize(self.index_path) == 0 or os.path.getsize(self.metadata_path) == 0:
            return
        try:
            import faiss
            from sentence_transformers import SentenceTransformer

            self.index = faiss.read_index(str(self.index_path))
            with self.metadata_path.open("rb") as file:
                metadata = pickle.load(file)
            self.metadata = metadata if isinstance(metadata, list) else []
            if getattr(self.index, "ntotal", len(self.metadata)) != len(self.metadata):
                self.index = None
                self.metadata = []
                return
            self.model = SentenceTransformer(self.model_name, local_files_only=True)
        except Exception:
            self.index = None
            self.metadata = []
            self.model = None

    @property
    def is_semantic_ready(self) -> bool:
        return self.index is not None and self.model is not None and bool(self.metadata)

    def reload(self) -> None:
        self.catalog = self._load_catalog()
        self.index = None
        self.metadata = []
        self.model = None
        self._load_index()

    def search(self, query: str, top_k: int = 10, requested_codes: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        query = clean_text(query)
        if not query or not self.catalog:
            return []
        profile = self._query_profile(query)
        codes = self._expanded_codes_for_query(list(requested_codes or infer_requested_type_codes(query)), profile)
        candidate_limit = max(top_k * 4, 30)
        expanded_query = profile["expanded_query"]

        if self.is_semantic_ready:
            # Merge semantic and deterministic candidates so FAISS recall is kept,
            # while exact role/skill matches can still win the final ranking.
            semantic_results = self._semantic_search(expanded_query, top_k=candidate_limit)
            keyword_results = self._keyword_search(expanded_query, top_k=candidate_limit)
            seed_results = self._role_seed_candidates(profile, top_k=candidate_limit)
            results = self._merge_results(semantic_results, keyword_results, seed_results)
        else:
            keyword_results = self._keyword_search(expanded_query, top_k=candidate_limit)
            seed_results = self._role_seed_candidates(profile, top_k=candidate_limit)
            results = self._merge_results(keyword_results, seed_results)

        reranked = self._apply_role_aware_scores(profile, results)
        filtered = [item for item in reranked if item_has_any_code(item, codes)]
        if len(filtered) < min(top_k, 3):
            filtered = reranked
        filtered = self._drop_weak_technical_matches(profile, filtered)
        filtered = self._ensure_cognitive_supplement(profile, filtered, top_k)
        filtered = self._ensure_requested_supplements(profile, filtered, top_k)
        return self._dedupe_and_limit(filtered, top_k)

    def _query_profile(self, query: str) -> Dict[str, Any]:
        normalized = normalize_for_match(query)
        expanded_query = self._expanded_query_text(query)
        tokens = set(tokenize(query))
        technical_role = any(self._contains_term(normalized, term) for term in TECH_ROLE_TERMS)
        technical_role = technical_role or any(self._contains_term(normalized, term) for term in TECH_STACK_TERMS)
        backend_role = any(self._contains_term(normalized, term) for term in {"backend", "back end"})
        wants_cognitive = any(self._contains_term(normalized, term) for term in COGNITIVE_QUERY_TERMS)
        wants_personality = any(self._contains_term(normalized, term) for term in PERSONALITY_QUERY_TERMS)
        wants_communication = any(self._contains_term(normalized, term) for term in COMMUNICATION_QUERY_TERMS)
        wants_technical = technical_role or any(
            self._contains_term(normalized, term)
            for term in {"technical", "coding", "programming", "skills", "skill"}
        )
        return {
            "normalized": normalized,
            "expanded_query": expanded_query,
            "expanded_normalized": normalize_for_match(expanded_query),
            "tokens": tokens,
            "technical_role": technical_role,
            "backend_role": backend_role,
            "wants_cognitive": wants_cognitive,
            "wants_personality": wants_personality,
            "wants_communication": wants_communication,
            "wants_technical": wants_technical,
        }

    def _expanded_query_text(self, query: str) -> str:
        normalized = normalize_for_match(query)
        expansions = []
        for trigger, expansion in QUERY_EXPANSIONS.items():
            if self._contains_term(normalized, trigger):
                expansions.append(expansion)
        return clean_text(" ".join([query, *expansions]))

    @staticmethod
    def _expanded_codes_for_query(codes: List[str], profile: Dict[str, Any]) -> List[str]:
        expanded = set(codes)
        if profile["technical_role"] or profile["wants_technical"]:
            expanded.update({"A", "K", "S"})
        if profile["wants_personality"] or profile["wants_communication"]:
            expanded.update({"B", "C", "P"})
        if profile["wants_cognitive"]:
            expanded.add("A")
        return sorted(expanded)

    def _merge_results(self, *result_groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for group in result_groups:
            for item in group:
                key = normalize_for_match(item.get("name", ""))
                if not key:
                    continue
                if key not in merged:
                    merged[key] = dict(item)
                    continue
                existing = merged[key]
                existing_score = float(existing.get("_score", 0.0))
                new_score = float(item.get("_score", 0.0))
                if new_score > existing_score:
                    existing.update(dict(item))
                existing["_score"] = max(existing_score, new_score)
                sources = {source for source in [existing.get("_rank_source"), item.get("_rank_source")] if source}
                existing["_rank_source"] = "+".join(sorted(sources))
        return list(merged.values())

    def _role_seed_candidates(self, profile: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        if not (
            profile["technical_role"]
            or profile["wants_technical"]
            or profile["wants_cognitive"]
            or profile["wants_personality"]
            or profile["wants_communication"]
        ):
            return []

        seeds = []
        for position, item in enumerate(self.catalog):
            codes = set(parse_test_type_codes(item.get("test_type")))
            item_text = normalize_for_match(catalog_item_text(item))
            item_name = normalize_for_match(item.get("name", ""))
            score = 0.0

            if profile["technical_role"] or profile["wants_technical"]:
                has_software_terms = any(self._contains_term(item_text, term) for term in SOFTWARE_ITEM_TERMS)
                if codes.intersection({"K", "S"}) and has_software_terms:
                    score += 12.0
                if "A" in codes:
                    score += 7.0
                if has_software_terms:
                    score += 5.0
                if profile["backend_role"] and any(self._contains_term(item_text, term) for term in BACKEND_ITEM_TERMS):
                    score += 3.0
                if self._product_key(item) in STRATEGIC_TECH_PRODUCTS:
                    score += 8.0

            if profile["wants_cognitive"] and "A" in codes:
                score += 12.0 if self._product_key(item) in STRATEGIC_TECH_PRODUCTS else 2.0
            if profile["wants_personality"] and "P" in codes:
                score += 4.0 if profile["technical_role"] else 9.0
                if self._product_key(item) in CORE_PERSONALITY_PRODUCTS:
                    score += 8.0
            if profile["wants_communication"] and codes.intersection({"B", "C", "P"}):
                score += 3.0

            if score > 0:
                copied = dict(item)
                copied["_score"] = score
                copied["_original_position"] = position
                copied["_rank_source"] = "role_seed"
                seeds.append(copied)

        seeds.sort(key=lambda item: (-item.get("_score", 0.0), item.get("_original_position", 999999), item.get("name", "")))
        return seeds[:top_k]

    def _semantic_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        try:
            vector = self.model.encode([query], normalize_embeddings=True)
            scores, indices = self.index.search(vector, min(top_k, len(self.metadata)))
        except Exception:
            return self._keyword_search(query, top_k)

        results = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0 or index >= len(self.metadata):
                continue
            item = dict(self.metadata[index])
            item["_score"] = float(score)
            item["_rank_source"] = "semantic"
            results.append(item)

        return self._rerank_with_keywords(query, results)

    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scored = []
        for position, item in enumerate(self.catalog):
            name_norm = normalize_for_match(item.get("name", ""))
            description_norm = normalize_for_match(item.get("description", ""))
            category_norm = normalize_for_match(f"{item.get('category', '')} {item.get('test_type', '')}")
            text_norm = normalize_for_match(catalog_item_text(item))
            name_tokens = set(tokenize(item.get("name", "")))
            description_tokens = set(tokenize(item.get("description", "")))
            category_tokens = set(tokenize(f"{item.get('category', '')} {item.get('test_type', '')}"))

            score = 0.0
            for token in query_tokens:
                if token in name_tokens:
                    score += 7.0
                elif token in description_tokens:
                    score += 3.0
                elif token in category_tokens:
                    score += 1.5
                elif token in text_norm:
                    score += 0.75

                if token in name_norm:
                    score += 1.5
                elif token in description_norm:
                    score += 0.5

            for term, weight in TECH_STACK_TERMS.items():
                if self._contains_term(query, term) and self._contains_term(name_norm, term):
                    score += weight
                elif self._contains_term(query, term) and self._contains_term(description_norm, term):
                    score += weight * 0.45

            if score > 0:
                copied = dict(item)
                copied["_score"] = float(score)
                copied["_original_position"] = position
                copied["_rank_source"] = "keyword"
                scored.append(copied)
        scored.sort(key=lambda item: (-item.get("_score", 0), item.get("_original_position", 999999), item.get("name", "")))
        return scored[:top_k]

    def _rerank_with_keywords(self, query: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return items
        reranked = []
        for item in items:
            text_norm = normalize_for_match(catalog_item_text(item))
            name_norm = normalize_for_match(item.get("name", ""))
            bonus = 0.0
            for token in query_tokens:
                if token in name_norm:
                    bonus += 0.08
                elif token in text_norm:
                    bonus += 0.03
            item = dict(item)
            item["_score"] = float(item.get("_score", 0.0)) + bonus
            reranked.append(item)
        reranked.sort(key=lambda item: (-item.get("_score", 0.0), item.get("name", "")))
        return reranked

    def _apply_role_aware_scores(self, profile: Dict[str, Any], items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        scored = []
        for position, item in enumerate(items):
            copied = dict(item)
            base_score = self._normalized_base_score(copied)
            role_bonus = self._role_bonus(profile, copied)
            copied["_score"] = base_score + role_bonus
            copied["_role_bonus"] = role_bonus
            copied.setdefault("_original_position", position)
            scored.append(copied)

        scored.sort(
            key=lambda item: (
                -item.get("_score", 0.0),
                item.get("_original_position", 999999),
                item.get("name", ""),
            )
        )
        return scored

    @staticmethod
    def _normalized_base_score(item: Dict[str, Any]) -> float:
        score = float(item.get("_score", 0.0))
        source = str(item.get("_rank_source", ""))
        if "semantic" in source and score <= 1.5:
            return score * 25.0
        return score

    def _role_bonus(self, profile: Dict[str, Any], item: Dict[str, Any]) -> float:
        codes = set(parse_test_type_codes(item.get("test_type")))
        item_text = normalize_for_match(catalog_item_text(item))
        item_name = normalize_for_match(item.get("name", ""))
        bonus = 0.0

        exact_technical_match = False
        for term, weight in TECH_STACK_TERMS.items():
            if not self._contains_term(profile["expanded_normalized"], term):
                continue
            if self._contains_term(item_name, term):
                bonus += weight
                exact_technical_match = True
            elif self._contains_term(item_text, term):
                bonus += weight * 0.45
                exact_technical_match = True

        has_technical_code = bool(codes.intersection({"K", "S"}))
        has_cognitive_code = "A" in codes
        has_personality_code = "P" in codes
        has_behavior_code = bool(codes.intersection({"B", "C"}))
        has_software_terms = any(self._contains_term(item_text, term) for term in SOFTWARE_ITEM_TERMS)
        has_sales_service_terms = any(self._contains_term(item_text, term) for term in SALES_SERVICE_ITEM_TERMS)
        is_job_solution_like = self._is_job_solution_like(item)

        if profile["technical_role"] or profile["wants_technical"]:
            # Software roles prefer K/S technical tests and A cognitive tests.
            # Personality and broad job bundles are held back unless explicitly needed.
            if has_technical_code and (has_software_terms or exact_technical_match):
                bonus += 18.0
            elif has_technical_code:
                bonus -= 3.0
            if has_cognitive_code:
                bonus += 11.0
            if has_software_terms:
                bonus += 7.0
            if profile["backend_role"] and any(self._contains_term(item_text, term) for term in BACKEND_ITEM_TERMS):
                bonus += 4.0
            if has_personality_code and not (has_technical_code or has_cognitive_code):
                bonus -= 6.0 if profile["wants_personality"] else 14.0
            if has_behavior_code and not (has_technical_code or has_cognitive_code):
                bonus -= 3.0
            if has_sales_service_terms and not exact_technical_match:
                bonus -= 10.0
            if is_job_solution_like and not exact_technical_match:
                bonus -= 16.0
            if self._product_key(item) in STRATEGIC_TECH_PRODUCTS:
                bonus += 10.0
            bonus += self._language_fit_adjustment(profile, item_text)

        if profile["wants_cognitive"]:
            if has_cognitive_code:
                bonus += 16.0
                if is_job_solution_like and self._product_key(item) not in STRATEGIC_TECH_PRODUCTS:
                    bonus -= 18.0
            elif has_technical_code:
                bonus += 2.0
            elif has_personality_code:
                bonus -= 5.0

        if profile["wants_personality"]:
            if has_personality_code:
                bonus += 6.0 if profile["technical_role"] else 14.0
                if self._product_key(item) in CORE_PERSONALITY_PRODUCTS:
                    bonus += 10.0
            elif has_behavior_code:
                bonus += 4.0
        elif profile["technical_role"] and has_personality_code and not (has_technical_code or has_cognitive_code):
            bonus -= 8.0

        if profile["wants_communication"] and (has_personality_code or has_behavior_code):
            bonus += 3.0

        return bonus

    def _language_fit_adjustment(self, profile: Dict[str, Any], item_text: str) -> float:
        query = profile["normalized"]
        mentions_java = self._contains_term(query, "java") or self._contains_term(query, "spring")
        mentions_python = self._contains_term(query, "python") or self._contains_term(query, "django")
        item_is_java = self._contains_term(item_text, "java") or self._contains_term(item_text, "spring")
        item_is_python = self._contains_term(item_text, "python")
        item_is_dotnet = self._contains_term(item_text, ".net") or self._contains_term(item_text, "c#")

        # Exact language intent should beat general backend similarity.
        if mentions_python and item_is_java and not mentions_java:
            return -80.0
        if mentions_java and item_is_python and not mentions_python:
            return -60.0
        if (mentions_python or mentions_java) and item_is_dotnet and not self._contains_term(query, ".net") and not self._contains_term(query, "c#"):
            return -70.0
        return 0.0

    def _ensure_cognitive_supplement(
        self, profile: Dict[str, Any], items: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        if not (profile["technical_role"] or profile["wants_technical"] or profile["wants_cognitive"]):
            return items

        deduped = self._dedupe_without_limit(items)
        if any("A" in parse_test_type_codes(item.get("test_type")) for item in deduped[:top_k]):
            return items

        cognitive_candidate = None
        for item in deduped[top_k:]:
            if "A" not in parse_test_type_codes(item.get("test_type")):
                continue
            if self._is_job_solution_like(item) and self._product_key(item) not in STRATEGIC_TECH_PRODUCTS:
                continue
            cognitive_candidate = item
            break
        if cognitive_candidate is None:
            return items

        key = normalize_for_match(cognitive_candidate.get("name", ""))
        without_candidate = [item for item in items if normalize_for_match(item.get("name", "")) != key]
        insert_at = min(4, max(1, top_k - 1), len(without_candidate))
        return without_candidate[:insert_at] + [cognitive_candidate] + without_candidate[insert_at:]

    def _ensure_requested_supplements(
        self, profile: Dict[str, Any], items: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        if not profile["wants_personality"] or top_k <= 1:
            return items

        # P assessments are supplemental in technical searches: insert the best
        # clean personality item after the strongest technical/cognitive block.
        deduped = self._dedupe_without_limit(items)
        if any("P" in parse_test_type_codes(item.get("test_type")) for item in deduped[:top_k]):
            return items

        personality_candidate = None
        for item in deduped[top_k:]:
            if "P" in parse_test_type_codes(item.get("test_type")):
                if profile["technical_role"] and self._is_job_solution_like(item):
                    continue
                personality_candidate = item
                break
        if personality_candidate is None:
            for item in deduped:
                if "P" in parse_test_type_codes(item.get("test_type")):
                    if profile["technical_role"] and self._is_job_solution_like(item):
                        continue
                    personality_candidate = item
                    break
        if personality_candidate is None:
            return items

        key = normalize_for_match(personality_candidate.get("name", ""))
        without_candidate = [item for item in items if normalize_for_match(item.get("name", "")) != key]
        insert_at = 6 if profile["technical_role"] else 4
        insert_at = min(insert_at, max(1, top_k - 1), len(without_candidate))
        return without_candidate[:insert_at] + [personality_candidate] + without_candidate[insert_at:]

    @staticmethod
    def _drop_weak_technical_matches(profile: Dict[str, Any], items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not (profile["technical_role"] or profile["wants_technical"]):
            return items
        # Avoid padding backend/software results with unrelated K-type catalog rows.
        strong_items = [item for item in items if float(item.get("_score", 0.0)) >= 8.0]
        return strong_items or items

    @staticmethod
    def _contains_term(text: str, term: str) -> bool:
        normalized = normalize_for_match(text)
        normalized_term = normalize_for_match(term)
        if not normalized or not normalized_term:
            return False
        if " " in normalized_term:
            return normalized_term in normalized
        if normalized_term.startswith("."):
            return normalized_term in normalized
        pattern = rf"(?<![a-z0-9+#.]){re.escape(normalized_term)}(?![a-z0-9+#.])"
        return re.search(pattern, normalized) is not None

    @staticmethod
    def _is_job_solution_like(item: Dict[str, Any]) -> bool:
        text = normalize_for_match(f"{item.get('name', '')} {item.get('description', '')}")
        markers = {
            "job focused", "short form", "solution", "jfa", "entry level",
            "contact center", "cashier", "teller", "store manager",
            "branch manager", "banker", "supervisor 7", "manager 7",
            "professional 7", "customer service", "sales 7",
        }
        return any(marker in text for marker in markers)

    @staticmethod
    def _dedupe_without_limit(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        output = []
        for item in items:
            key = normalize_for_match(item.get("name", ""))
            if not key or key in seen:
                continue
            seen.add(key)
            output.append(item)
        return output

    @staticmethod
    def _dedupe_and_limit(items: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        seen = set()
        family_counts: Dict[str, int] = {}
        output = []
        for item in items:
            key = normalize_for_match(item.get("name", ""))
            if not key or key in seen:
                continue
            family_key = RetrievalEngine._family_key(item)
            # Duplicate suppression keeps one product family, such as .NET or Verify,
            # from crowding out more relevant complementary assessments.
            if family_counts.get(family_key, 0) >= 3:
                continue
            seen.add(key)
            family_counts[family_key] = family_counts.get(family_key, 0) + 1
            output.append(item)
            if len(output) >= top_k:
                break
        return output

    @staticmethod
    def _family_key(item: Dict[str, Any]) -> str:
        name = normalize_for_match(item.get("name", ""))
        if name.startswith(".net") or name.startswith("ado.net"):
            return "dotnet"
        if name.startswith("shl verify interactive"):
            return "verify interactive"
        if name.startswith("verify"):
            return "verify"
        tokens = name.split()
        return " ".join(tokens[:2]) if len(tokens) >= 2 else name

    @staticmethod
    def _product_key(item: Dict[str, Any]) -> str:
        return normalize_for_match(item.get("name", ""))

    def find_by_name(self, name: str, limit: int = 3) -> List[Dict[str, Any]]:
        target = normalize_for_match(name)
        if not target:
            return []
        scored: List[Tuple[float, int, Dict[str, Any]]] = []
        for position, item in enumerate(self.catalog):
            item_name = normalize_for_match(item.get("name", ""))
            if not item_name:
                continue
            ratio = SequenceMatcher(None, target, item_name).ratio()
            if target in item_name or item_name in target:
                ratio = max(ratio, 0.92)
            acronym = "".join(word[0] for word in item_name.split() if word)
            if target == acronym or target.replace(" ", "") == acronym:
                ratio = max(ratio, 0.95)
            if ratio >= 0.70:
                scored.append((ratio, position, item))
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [dict(item) for _, _, item in scored[:limit]]

    def extract_comparison_items(self, user_text: str, max_items: int = 4) -> Tuple[List[Dict[str, Any]], List[str]]:
        normalized = clean_text(user_text)
        separators = r"\b(?:and|vs\.?|versus|between|compare|difference between|differences between|with)\b|,|/"
        parts = [clean_text(part) for part in re_split(separators, normalized) if clean_text(part)]
        candidates = []
        for part in parts:
            cleaned = re_cleanup_compare_part(part)
            if cleaned and len(cleaned) <= 80:
                candidates.append(cleaned)
        if not candidates:
            candidates = [normalized]

        matched = []
        missing = []
        seen = set()
        for candidate in candidates:
            matches = self.find_by_name(candidate, limit=1)
            if matches:
                item = matches[0]
                key = normalize_for_match(item.get("name"))
                if key not in seen:
                    seen.add(key)
                    matched.append(item)
            elif len(candidate.split()) <= 6:
                missing.append(candidate)
            if len(matched) >= max_items:
                break

        if missing:
            return matched[:max_items], missing

        if len(matched) < 2:
            search_matches = self.search(user_text, top_k=max_items)
            for item in search_matches:
                key = normalize_for_match(item.get("name"))
                if key and key not in seen:
                    seen.add(key)
                    matched.append(item)
                if len(matched) >= max_items:
                    break
        return matched[:max_items], missing

    @staticmethod
    def to_response_items(items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        return [format_recommendation(item) for item in items]


def re_split(pattern: str, text: str) -> List[str]:
    import re

    return re.split(pattern, text, flags=re.IGNORECASE)


def re_cleanup_compare_part(text: str) -> str:
    import re

    cleaned = clean_text(text)
    cleaned = re.sub(r"^(what is|what are|the|a|an|is|are|please|can you|tell me)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(difference|differences|compare|comparison|better)\b", "", cleaned, flags=re.IGNORECASE)
    return clean_text(cleaned.strip(" ?:."))
