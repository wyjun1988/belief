#!/usr/bin/env python3
"""OWLv2 후처리 API 호환 (2026-09-22, 프로6000 보고).

transformers 신판에서 `Owlv2Processor.post_process_object_detection` 이 빠지고
`post_process_grounded_object_detection` 만 남았다(M2 4.57.6 은 둘 다 있고, RTX 환경은 신판만).
반환 키도 판본마다 다르다(labels / text_labels). 호출부는 이 함수만 쓴다.

    from owl_compat import owl_post
    res = owl_post(processor, outputs, threshold=0.05, target_sizes=ts)   # [{boxes, scores, labels}, ...]
"""
def owl_post(processor, outputs, threshold=0.0, target_sizes=None):
    fn = getattr(processor, "post_process_grounded_object_detection", None)
    if fn is None: fn = getattr(processor, "post_process_object_detection")
    try:
        res = fn(outputs=outputs, threshold=threshold, target_sizes=target_sizes)
    except TypeError:                      # 구판 위치인자 판본
        res = fn(outputs, threshold, target_sizes)
    out = []
    for r in res:
        d = dict(r)
        if "labels" not in d and "text_labels" in d: d["labels"] = d["text_labels"]
        out.append(d)
    return out
