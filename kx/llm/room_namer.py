"""작은 LLM 으로 **물체 이름 → 방** 사전을 만든다. 기하를 전혀 쓰지 않는다.

지금 방 서명의 사전은 v1 그래프의 3D 위치를 구역지도에 넣어 만든다. 그래서 포즈가
나쁘면 사전이 통째로 틀어진다(graph_sam 0.971 → graph_allmodel 0.673). 물체 **이름**만으로
사전을 만들 수 있으면 그 병목이 사라진다 — 상식은 기하가 없어도 아는 것이기 때문이다.

⚠️ 0.5B 급에서 **자유 생성은 못 쓴다.** "refrigerator 는 어느 방?" 에 "Freezer" 라고
답한다. 대신 후보 방을 주고 **각 후보의 로그확률을 재서 argmax** 를 고른다. 생성이
아니라 분류라 형식 붕괴가 없고 온도·샘플링 변수도 없다.

⚠️ 그대로 쓰면 **라벨 사전확률에 쏠린다** — 0.5B 는 소파·커피테이블·변기까지 전부
bedroom 이라 답했다. 중립 프롬프트("a thing")에서의 로그확률을 빼서 상쇄한다
(contextual calibration). 물체가 주는 **증거만** 남기는 것이다.
"""
import os
import re

import numpy as np

DEFAULT_MODEL = os.environ.get("KX_LLM", "Qwen/Qwen2.5-0.5B-Instruct")
PROMPT = ("In a typical home, which room would you find a {obj}?\n"
          "Answer with exactly one of: {opts}.\nAnswer:")


def _pretty(name):
    """`BlackCoffeeTable` / `Fridge_2` → `black coffee table` / `fridge`."""
    s = re.sub(r"[_\d]+", " ", str(name))
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)
    return " ".join(s.split()).lower().strip()


class RoomNamer:
    def __init__(self, rooms, model=DEFAULT_MODEL, device="cpu", batch=16):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.rooms = list(rooms)
        self.device = device
        self.batch = batch
        self.tok = AutoTokenizer.from_pretrained(model)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.mdl = AutoModelForCausalLM.from_pretrained(
            model, dtype=torch.float32, use_safetensors=True).eval().to(device)
        self.opts = ", ".join(r.replace("_", " ") for r in self.rooms)
        # 후보 답의 토큰열을 미리 만들어 둔다 (앞 공백 포함이 자연스럽다)
        self.cand = [self.tok(" " + r.replace("_", " "), add_special_tokens=False)["input_ids"]
                     for r in self.rooms]
        self.bias = self._raw("thing")          # 중립 기준선 — 이걸 빼면 라벨 편향이 준다

    def _prompt_ids(self, obj):
        msg = [{"role": "user", "content": PROMPT.format(obj=_pretty(obj), opts=self.opts)}]
        return self.tok.apply_chat_template(msg, add_generation_prompt=True)

    def _raw(self, obj):
        """보정 전 평균 로그확률."""
        torch = self.torch
        base = self._prompt_ids(obj)
        out = np.zeros(len(self.rooms))
        for k, c in enumerate(self.cand):
            ids = torch.tensor([base + c], device=self.device)
            with torch.no_grad():
                logits = self.mdl(ids).logits[0]
            lp = torch.log_softmax(logits[:-1], -1)
            tgt = ids[0, 1:]
            sel = lp[torch.arange(len(tgt)), tgt][len(base) - 1:]
            out[k] = float(sel.mean())          # 길이 정규화 — 긴 방 이름 불이익 제거
        return out

    def score(self, obj):
        return self._raw(obj) - self.bias

    def label(self, obj, margin_min=0.0):
        s = self.score(obj)
        o = np.argsort(-s)
        if s[o[0]] - s[o[1]] < margin_min:
            return None, float(s[o[0]] - s[o[1]])
        return self.rooms[o[0]], float(s[o[0]] - s[o[1]])

    def label_many(self, objs, margin_min=0.0):
        return {o: self.label(o, margin_min) for o in objs}
