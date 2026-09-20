"""타입 키 → 검출기/제로샷 프롬프트용 문장 별칭 (2026-09-20 §166-94).

  from owl_alias import alias
  sp(t) 안에서 t 대신 alias(t) 를 쓴다. 키(gt0·vocab·initmap 의 type)는 절대 바꾸지 않는다 — 조인이 깨진다.
근거·항목은 data/owl_alias.json. 없거나 비어 있으면 무변환.
"""
import json, os
_P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "owl_alias.json")
try: _A = {k: v for k, v in json.load(open(_P)).items() if not k.startswith("_")}
except Exception: _A = {}
if os.environ.get("OWL_ALIAS", "1") != "1": _A = {}     # OWL_ALIAS=0 으로 끈다(대조용)
def alias(t: str) -> str:
    return _A.get(t, _A.get(str(t).replace("_", " ").lower(), t))
