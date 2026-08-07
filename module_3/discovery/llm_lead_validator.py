from __future__ import annotations

"""
Provider-neutral LLM validator for shortlisted lead candidates.

The validator receives evidence collected and filtered by the discovery
pipeline. It does not browse URLs or perform resource collection.
"""

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence


logger = logging.getLogger(__name__)


class LeadValidationDecision(str, Enum):
    VALID_LEAD = "valid_lead"
    NOT_A_LEAD = "not_a_lead"
    MANUAL_REVIEW = "manual_review"


@dataclass(slots=True)
class LeadValidationCandidate:
    candidate_id: str
    title: str
    url: str
    snippet: str = ""
    content_excerpt: str = ""

    preliminary_company: str | None = None
    preliminary_signal_type: str | None = None
    preliminary_confidence: float = 0.0

    matched_services: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    uncertainty_reasons: list[str] = field(default_factory=list)

    # --- from friend's branch: deadline awareness ---
    deadline_status: str = "unknown"
    deadline: str | None = None
    deadline_reason: str = ""
    deadline_confidence: float = 0.0

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "content_excerpt": self.content_excerpt,
            "preliminary_company": self.preliminary_company,
            "preliminary_signal_type": self.preliminary_signal_type,
            "preliminary_confidence": round(
                max(0.0, min(1.0, float(self.preliminary_confidence))),
                4,
            ),
            "matched_services": self.matched_services,
            "evidence": self.evidence,
            "uncertainty_reasons": self.uncertainty_reasons,
            "deadline_status": self.deadline_status,
            "deadline": self.deadline,
            "deadline_reason": self.deadline_reason,
            "deadline_confidence": round(
                max(0.0, min(1.0, float(self.deadline_confidence))),
                4,
            ),
        }


@dataclass(slots=True)
class LeadValidationResult:
    candidate_id: str
    decision: LeadValidationDecision

    buyer_organization: str | None = None
    lead_type: str | None = None
    matched_service_ids: list[str] = field(default_factory=list)

    requires_external_supplier: bool = False
    supplier_already_selected: bool = False
    is_current: bool = False

    confidence: float = 0.0
    reason: str = ""
    validation_source: str = "claude"
    # --- from your branch: country extraction, used by discover()'s
    # country-override logic ---
    country: str | None = None
    buyer_country_raw: str | None = None
    canonical_country: str | None = None
    region_status: str = "unknown"
    region_confidence: float = 0.0
    region_evidence: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["decision"] = self.decision.value
        return data


class LLMLeadValidatorError(RuntimeError):
    """Base exception for LLM validator failures."""


class LLMQuotaExceededError(LLMLeadValidatorError):
    """Raised when the LLM provider rejects a request due to quota or rate limits."""


class LLMResponseFormatError(LLMLeadValidatorError):
    """Raised when the LLM does not return valid structured output."""


class LLMLeadValidator:
    """Batch validator for lead candidates.

    The validator does not browse URLs. It evaluates only the evidence supplied
    in each LeadValidationCandidate.

    The model object may be:

    1. An adapter exposing ``generate_content(prompt)``.
    2. An adapter exposing ``generate(prompt)``.
    3. A callable accepting one prompt string.

    The response must either be a string or expose a ``text`` attribute.
    """

    SYSTEM_INSTRUCTIONS_TEMPLATE = """
The objective is to determine whether the BUYER ORGANIZATION represents a genuine sales opportunity.

Ignore information describing:

- the publisher;
- the website owner;
- the tender portal;
- the search engine result;
- intermediary procurement websites.

Evaluate only the underlying buyer opportunity.

You are validating enterprise technology sales leads for Triway.

Evaluate each candidate independently using only the supplied evidence.
Do not browse the URL and do not invent missing facts.

A VALID_LEAD must satisfy all of the following:
1. A real buyer organisation is identifiable.
2. The evidence describes a current or upcoming technology requirement,
   initiative, procurement, budget signal, implementation programme,
   partner search, or strategic transformation event.
3. The requirement relates to at least one supplied Triway service.
4. There is a realistic possibility that an external supplier could help.
5. The page is not merely generic educational content, vendor marketing,
   a case study, research, a webinar, social content, a training page,
   a job aggregator, or a service-provider landing page.
6. The contract has not already been awarded and the project is not completed,
   expired, cancelled, or clearly historical.

Use:
- valid_lead: evidence is sufficient and all conditions are met.
- not_a_lead: evidence clearly shows it is not an actionable buyer opportunity.
- manual_review: evidence is incomplete, ambiguous, inaccessible, or too weak
  to make a reliable decision.

Important rules:
- The local similarity score is supporting evidence only. A low similarity
  score must not automatically cause rejection when the requirement clearly
  matches a supplied Triway service.
- Only include matched_service_ids directly supported by the supplied evidence.
  Return an empty list when no supplied Triway service can be justified.
- Determine whether the evidence describes one specific opportunity. A search
  page, category page, tender directory, multi-opportunity listing, or generic
  tender landing page is not a valid lead.
- Use the supplied deadline analysis as the primary evidence for temporal status.

  * If deadline_status is "expired", return not_a_lead. Never override it.
  * If deadline_status is "open", you may consider the opportunity current.
  * If deadline_status is "upcoming", treat it as a valid future opportunity.
  * If deadline_status is "unknown", inspect only the supplied document evidence.

  Never infer that an opportunity is current solely because of:
  - the URL;
  - the PDF filename;
  - upload or publication dates;
  - directory names;
  - page timestamps.

  Only explicit submission deadlines, closing dates, procurement schedules,
  or clear statements that the opportunity is accepting responses may be used
  as evidence that it is current.

  If currentness cannot be verified, return manual_review rather than valid_lead.
- A technical job vacancy by itself is normally not enough.
- A procurement article or newsletter mentioning procurement is not itself
  a procurement lead.
- A vendor announcing its own product or service is not a buyer lead.
- A contract award is not an open lead.
- Set supplier_already_selected to true only when the evidence explicitly says
  that the contract has been awarded or a supplier has already been selected.
- Pre-solicitation intelligence may be a valid lead when a buyer, initiative,
  budget, or active planning signal is clearly identified.
- Never invent company names, dates, contacts, procurement status, services,
  supplier selection, or buying intent.
- Keep reason under 40 words and evidence-based.
- Confidence reflects evidence quality, not opportunity importance.

When buyer organization, region, and procurement status conflict across inputs,
prioritize the fetched procurement document content over the search snippet,
page title, URL, or website metadata.

REGION IDENTIFICATION

Determine only the country of the buyer organization or contracting authority.

Do not use:

- supplier locations;
- website hosting location;
- contact addresses unless they clearly belong to the buyer;
- a contact person's nationality;
- the location of a third-party tender aggregator;
- example locations;
- delivery locations unless they clearly identify the buyer country.

Allowed service regions:

{allowed_regions_text}

Return:

buyer_country_raw
The location exactly as written in the supplied evidence, or null.

canonical_country
Exactly one value copied from the allowed-region list when the buyer country is
supported. Otherwise return null.

region_status
One of: supported, unsupported, unknown.

region_confidence
A number from 0 to 1 representing evidence quality.

region_evidence
Exact supporting evidence from the supplied source, or null.

Region rules:
- Region eligibility is a hard business rule.
- Use supported only when the buyer country is clearly established and appears
  in the allowed-region list.
- Use unsupported only when the buyer country is clearly established and falls
  outside the allowed-region list.
- Use unknown when the buyer country cannot be reliably established.
- If region_status is unsupported, decision MUST be not_a_lead.
  Never return manual_review or valid_lead for an unsupported buyer country.
- If region_status is unknown, decision MUST NOT be valid_lead.
  Use manual_review unless another clear rejection reason already makes the
  candidate not_a_lead.
- A candidate may be valid_lead only when region_status is supported.
- When supported, copy canonical_country exactly from the allowed-region list.
- Never abbreviate canonical_country.
- Never invent a new canonical country value.
- Never guess.

Return JSON only using this exact top-level structure:
{{
  "results": [
    {{
      "candidate_id": "string",
      "decision": "valid_lead | not_a_lead | manual_review",
      "buyer_organization": "string or null",
      "lead_type": "string or null",
      "matched_service_ids": ["string"],
      "requires_external_supplier": true,
      "supplier_already_selected": false,
      "is_current": true,
      "confidence": 0.0,
      "reason": "string",
      "buyer_country_raw": "string or null",
      "canonical_country": "string or null",
      "region_status": "supported | unsupported | unknown",
      "region_confidence": 0.0,
      "region_evidence": "string or null"
    }}
  ]
}}

Do not include explanations outside the JSON.
Do not include Markdown or code fences.
Every candidate must produce exactly one result.
Every field in the schema is required.
Use null for unknown string values and [] for no matched services.
""".strip()

    def __init__(
        self,
        model: Any,
        *,
        batch_size: int = 8,
        max_candidates: int | None = 20,
        max_excerpt_chars: int = 3000,
        max_retries: int = 3,
        retry_base_seconds: float = 2.0,
        raise_on_batch_failure: bool = False,
        allowed_regions: list[str] | None = None,
    ) -> None:
        if model is None:
            raise ValueError("An LLM model or model adapter is required.")

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        if max_candidates is not None and max_candidates <= 0:
            raise ValueError(
                "max_candidates must be greater than zero or None."
            )

        if max_excerpt_chars <= 0:
            raise ValueError("max_excerpt_chars must be greater than zero.")

        if max_retries < 0:
            raise ValueError("max_retries cannot be negative.")

        if retry_base_seconds <= 0:
            raise ValueError("retry_base_seconds must be greater than zero.")

        self.model = model
        self.batch_size = batch_size
        self.max_candidates = max_candidates
        self.max_excerpt_chars = max_excerpt_chars
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.raise_on_batch_failure = raise_on_batch_failure
        self.allowed_regions = sorted(
            {
                str(region).strip()
                for region in (allowed_regions or [])
                if str(region).strip()
            }
        )

    def validate_candidates(
        self,
        candidates: Sequence[LeadValidationCandidate],
    ) -> list[LeadValidationResult]:
        """Validate shortlisted candidates in small LLM batches."""

        if self.max_candidates is None:
            limited_candidates = list(candidates)
        else:
            limited_candidates = list(candidates[: self.max_candidates])

        if not limited_candidates:
            return []

        results: list[LeadValidationResult] = []

        for batch in self._batched(
            limited_candidates,
            self.batch_size,
        ):
            try:
                results.extend(self._validate_batch(batch))
            except LLMQuotaExceededError:
                logger.warning(
                    "LLM quota exceeded. Remaining candidates are marked "
                    "for manual review."
                )

                unresolved_ids = {
                    candidate.candidate_id
                    for candidate in limited_candidates
                } - {
                    result.candidate_id
                    for result in results
                }

                results.extend(
                    self._manual_review_result(
                        candidate_id=candidate_id,
                        reason="LLM validation was unavailable because quota was exceeded.",
                    )
                    for candidate_id in unresolved_ids
                )
                break

            except LLMLeadValidatorError as exc:
                logger.warning(
                    "LLM batch validation failed: %s",
                    exc,
                )

                if self.raise_on_batch_failure:
                    raise

                results.extend(
                    self._manual_review_result(
                        candidate_id=candidate.candidate_id,
                        reason="LLM returned an unusable validation response.",
                    )
                    for candidate in batch
                )

            except Exception as exc:
                logger.exception(
                    "Unexpected LLM validation error."
                )

                if self.raise_on_batch_failure:
                    raise LLMLeadValidatorError(str(exc)) from exc

                results.extend(
                    self._manual_review_result(
                        candidate_id=candidate.candidate_id,
                        reason="LLM validation failed unexpectedly.",
                    )
                    for candidate in batch
                )

        return self._restore_input_order(
            candidates=limited_candidates,
            results=results,
        )

    def _validate_batch(
        self,
        candidates: Sequence[LeadValidationCandidate],
    ) -> list[LeadValidationResult]:
        prompt = self._build_prompt(candidates)

        # --- from friend's branch: retry with backoff on transient errors ---
        response = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self._call_model(prompt)
                break
            except Exception as exc:
                if self._looks_like_quota_error(exc):
                    raise LLMQuotaExceededError(str(exc)) from exc

                if (
                    self._looks_like_transient_error(exc)
                    and attempt < self.max_retries
                ):
                    delay = self.retry_base_seconds * (2 ** attempt)
                    logger.warning(
                        "Transient LLM failure. Attempt %s/%s. "
                        "Retrying in %.1fs. Error: %s",
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
                    continue

                raise LLMLeadValidatorError(
                    f"LLM request failed: {exc}"
                ) from exc

        if response is None:
            raise LLMLeadValidatorError(
                "LLM returned no response after retries."
            )

        response_text = self._extract_response_text(response)

        logger.debug(
            "Raw LLM response (%s chars): %s",
            len(response_text),
            self._truncate(response_text, 4000),
        )

        try:
            payload = self._parse_json_response(response_text)
        except LLMResponseFormatError:
            logger.warning(
                "Unable to parse LLM response. Raw response: %s",
                self._truncate(response_text, 4000),
            )
            raise

        raw_results = payload.get("results")

        if not isinstance(raw_results, list):
            raise LLMResponseFormatError(
                "LLM response did not contain a results list."
            )

        expected_ids = {
            candidate.candidate_id
            for candidate in candidates
        }

        parsed_results: list[LeadValidationResult] = []
        returned_ids: set[str] = set()

        for item in raw_results:
            if not isinstance(item, dict):
                continue

            result = self._parse_result(item)

            if result.candidate_id not in expected_ids:
                logger.warning(
                    "Ignoring LLM result with unknown candidate_id: %s",
                    result.candidate_id,
                )
                continue

            if result.candidate_id in returned_ids:
                logger.warning(
                    "Ignoring duplicate LLM result for candidate_id: %s",
                    result.candidate_id,
                )
                continue

            returned_ids.add(result.candidate_id)
            parsed_results.append(result)

        missing_ids = expected_ids - returned_ids

        parsed_results.extend(
            self._manual_review_result(
                candidate_id=candidate_id,
                reason="LLM did not return a decision for this candidate.",
            )
            for candidate_id in missing_ids
        )

        return parsed_results

    def _build_prompt(
        self,
        candidates: Sequence[LeadValidationCandidate],
    ) -> str:
        prepared_candidates: list[dict[str, Any]] = []

        for candidate in candidates:
            payload = candidate.to_prompt_dict()
            payload["content_excerpt"] = self._truncate(
                payload.get("content_excerpt", ""),
                self.max_excerpt_chars,
            )
            payload["snippet"] = self._truncate(
                payload.get("snippet", ""),
                700,
            )
            payload["evidence"] = [
                self._truncate(str(value), 500)
                for value in payload.get("evidence", [])[:8]
            ]
            payload["uncertainty_reasons"] = [
                self._truncate(str(value), 300)
                for value in payload.get(
                    "uncertainty_reasons",
                    [],
                )[:6]
            ]
            payload["matched_services"] = payload.get(
                "matched_services",
                [],
            )[:5]

            prepared_candidates.append(payload)

        candidate_json = json.dumps(
            {"candidates": prepared_candidates},
            ensure_ascii=False,
            separators=(",", ":"),
        )

        allowed_regions_text = "\n".join(
            f"- {region}"
            for region in self.allowed_regions
        ) or "- None configured"

        system_instructions = self.SYSTEM_INSTRUCTIONS_TEMPLATE.format(
            allowed_regions_text=allowed_regions_text,
        )

        return (
            f"{system_instructions}\n\n"
            "Candidates to validate:\n"
            f"{candidate_json}"
        )

    def _call_model(self, prompt: str) -> Any:
        """Call the configured LLM adapter."""
        if hasattr(self.model, "generate_content"):
            return self.model.generate_content(prompt)
        if hasattr(self.model, "generate"):
            return self.model.generate(prompt)
        if callable(self.model):
            return self.model(prompt)
        raise LLMLeadValidatorError(
            "Unsupported LLM model object. Expected generate_content(), "
            "generate(), or a callable."
        )

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        if isinstance(response, str):
            text = response
        else:
            text = getattr(response, "text", None)

        if not isinstance(text, str) or not text.strip():
            raise LLMResponseFormatError(
                "LLM response did not contain text."
            )

        return text.strip()

    @staticmethod
    def _parse_json_response(text: str) -> dict[str, Any]:
        cleaned = text.strip()

        cleaned = re.sub(
            r"^(?:here(?:'s| is)\s+(?:the\s+)?(?:json|result|response)s?\s*:?|"
            r"response\s*:|json\s*:|analysis\s*:)",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

        # Remove optional Markdown JSON fences.
        if cleaned.startswith("```"):
            cleaned = re.sub(
                r"^```(?:json)?\s*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
            cleaned = re.sub(
                r"\s*```$",
                "",
                cleaned,
            )

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            # Last-resort extraction if the model adds short prose around JSON.
            start = cleaned.find("{")
            end = cleaned.rfind("}")

            if start == -1 or end == -1 or end <= start:
                raise LLMResponseFormatError(
                    "LLM response was not valid JSON."
                )

            try:
                payload = json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError as exc:
                raise LLMResponseFormatError(
                    "LLM response was not valid JSON."
                ) from exc

        if not isinstance(payload, dict):
            raise LLMResponseFormatError(
                "LLM response must be a JSON object."
            )

        return payload

    @classmethod
    def _parse_result(
        cls,
        item: dict[str, Any],
    ) -> LeadValidationResult:
        candidate_id = str(
            item.get("candidate_id", "")
        ).strip()

        if not candidate_id:
            raise LLMResponseFormatError(
                "LLM result is missing candidate_id."
            )

        decision_raw = str(
            item.get("decision", "")
        ).strip().lower()

        try:
            decision = LeadValidationDecision(decision_raw)
        except ValueError:
            decision = LeadValidationDecision.MANUAL_REVIEW

        matched_service_ids_raw = item.get(
            "matched_service_ids",
            [],
        )

        if not isinstance(matched_service_ids_raw, list):
            matched_service_ids_raw = []

        matched_service_ids = [
            str(value).strip()
            for value in matched_service_ids_raw
            if str(value).strip()
        ]

        confidence = cls._safe_float(
            item.get("confidence", 0.0)
        )
        confidence = max(0.0, min(1.0, confidence))

        buyer_organization = cls._optional_string(
            item.get("buyer_organization")
        )
        lead_type = cls._optional_string(
            item.get("lead_type")
        )

        reason = str(
            item.get("reason", "")
        ).strip()

        if not reason:
            reason = "No validation reason was returned."

        buyer_country_raw = cls._optional_string(
            item.get("buyer_country_raw")
        )
        canonical_country = cls._optional_string(
            item.get("canonical_country")
        )

        region_status = str(
            item.get("region_status", "unknown")
        ).strip().lower()

        if region_status not in {
            "supported",
            "unsupported",
            "unknown",
        }:
            region_status = "unknown"

        region_confidence = cls._safe_float(
            item.get("region_confidence", 0.0)
        )
        region_confidence = max(
            0.0,
            min(1.0, region_confidence),
        )

        result = LeadValidationResult(
            candidate_id=candidate_id,
            decision=decision,
            buyer_organization=buyer_organization,
            lead_type=lead_type,
            matched_service_ids=matched_service_ids,
            requires_external_supplier=cls._safe_bool(
                item.get("requires_external_supplier", False)
            ),
            supplier_already_selected=cls._safe_bool(
                item.get("supplier_already_selected", False)
            ),
            is_current=cls._safe_bool(
                item.get("is_current", False)
            ),
            confidence=confidence,
            reason=reason,
            # Keep the legacy country field populated so existing downstream
            # country-override logic continues to work without modification.
            country=canonical_country,
            buyer_country_raw=buyer_country_raw,
            canonical_country=canonical_country,
            region_status=region_status,
            region_confidence=region_confidence,
            region_evidence=cls._optional_string(
                item.get("region_evidence")
            ),
        )

        return cls._apply_safety_consistency(result)

    @staticmethod
    def _apply_safety_consistency(
        result: LeadValidationResult,
    ) -> LeadValidationResult:
        """Apply hard policy rules and prevent contradictory LLM output."""

        # Region eligibility is a hard business rule. A clearly identified
        # buyer outside the allowed regions is rejected regardless of the
        # model's original decision.
        if result.region_status == "unsupported":
            result.decision = LeadValidationDecision.NOT_A_LEAD
            region_reason = (
                "The buyer country is outside the allowed service regions."
            )
            if region_reason.casefold() not in result.reason.casefold():
                result.reason = (
                    f"{result.reason} Policy enforcement: {region_reason}"
                ).strip()
            result.confidence = max(result.confidence, result.region_confidence)
            return result

        # An unknown buyer country cannot be accepted as a valid lead.
        if (
            result.region_status == "unknown"
            and result.decision == LeadValidationDecision.VALID_LEAD
        ):
            result.decision = LeadValidationDecision.MANUAL_REVIEW
            result.reason = (
                result.reason
                + " Policy enforcement: The buyer country could not be "
                  "established reliably."
            )
            result.confidence = min(result.confidence, 0.59)
            return result

        # A supported result must include the canonical country copied from
        # the allowed-region list.
        if (
            result.region_status == "supported"
            and not result.canonical_country
            and result.decision == LeadValidationDecision.VALID_LEAD
        ):
            result.decision = LeadValidationDecision.MANUAL_REVIEW
            result.reason = (
                result.reason
                + " Policy enforcement: The region was marked supported "
                  "without a canonical country."
            )
            result.confidence = min(result.confidence, 0.59)
            return result

        if result.decision != LeadValidationDecision.VALID_LEAD:
            return result

        contradiction_reasons: list[str] = []

        if not result.buyer_organization:
            contradiction_reasons.append(
                "No buyer organisation was identified."
            )

        if not result.is_current:
            contradiction_reasons.append(
                "The opportunity was not confirmed as current."
            )

        if result.supplier_already_selected:
            contradiction_reasons.append(
                "A supplier appears to have already been selected."
            )

        if not result.matched_service_ids:
            contradiction_reasons.append(
                "No Triway service was matched."
            )

        if contradiction_reasons:
            result.decision = LeadValidationDecision.MANUAL_REVIEW
            result.reason = (
                result.reason
                + " Validator consistency check: "
                + " ".join(contradiction_reasons)
            )
            result.confidence = min(result.confidence, 0.59)

        return result

    @staticmethod
    def _manual_review_result(
        *,
        candidate_id: str,
        reason: str,
    ) -> LeadValidationResult:
        return LeadValidationResult(
            candidate_id=candidate_id,
            decision=LeadValidationDecision.MANUAL_REVIEW,
            confidence=0.0,
            reason=reason,
            validation_source="fallback",
        )

    @staticmethod
    def _restore_input_order(
        *,
        candidates: Sequence[LeadValidationCandidate],
        results: Sequence[LeadValidationResult],
    ) -> list[LeadValidationResult]:
        by_id = {
            result.candidate_id: result
            for result in results
        }

        ordered: list[LeadValidationResult] = []

        for candidate in candidates:
            ordered.append(
                by_id.get(
                    candidate.candidate_id,
                    LLMLeadValidator._manual_review_result(
                        candidate_id=candidate.candidate_id,
                        reason="No validation result was available.",
                    ),
                )
            )

        return ordered

    @staticmethod
    def _looks_like_quota_error(exc: Exception) -> bool:
        text = str(exc).casefold()

        quota_markers = (
            "429",
            "resource_exhausted",
            "quota exceeded",
            "rate limit",
            "too many requests",
        )

        return any(
            marker in text
            for marker in quota_markers
        )

    @staticmethod
    def _looks_like_transient_error(exc: Exception) -> bool:
        text = str(exc).casefold()

        transient_markers = (
            "503",
            "500",
            "502",
            "504",
            "unavailable",
            "service unavailable",
            "temporarily unavailable",
            "high demand",
            "deadline exceeded",
            "timeout",
            "timed out",
            "connection reset",
            "connection aborted",
            "internal server error",
        )

        return any(
            marker in text
            for marker in transient_markers
        )

    @staticmethod
    def _truncate(value: Any, limit: int) -> str:
        text = str(value or "").strip()

        if len(text) <= limit:
            return text

        return text[: limit - 3].rstrip() + "..."

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        if value is None:
            return None

        text = str(value).strip()

        if not text or text.casefold() in {"none", "null"}:
            return None

        return text

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _safe_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)):
            return bool(value)

        if isinstance(value, str):
            return value.strip().casefold() in {
                "true",
                "1",
                "yes",
                "y",
            }

        return False

    @staticmethod
    def _batched(
        values: Sequence[LeadValidationCandidate],
        size: int,
    ) -> Iterable[Sequence[LeadValidationCandidate]]:
        for start in range(0, len(values), size):
            yield values[start : start + size]