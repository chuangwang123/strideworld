"""Process-local inference-mode correction; never edit frozen official source."""
import json

def install(engine):
    original = engine.build_pipe
    def build(args):
        result = original(args)
        pipe, transformer, vae, device = result
        before = bool(transformer.training)
        for model in (transformer, vae, getattr(pipe, "text_encoder", None)):
            if model is not None:
                model.eval()
        if any(module.training for module in transformer.modules()):
            raise RuntimeError("Transformer subtree still in training mode")
        if not getattr(transformer, "_strideworld_eval_guard_installed", False):
            def require_eval(module, args):
                if any(child.training for child in module.modules()):
                    raise RuntimeError("Training mode reappeared before transformer forward")
            transformer.register_forward_pre_hook(require_eval)
            transformer._strideworld_eval_guard_installed = True
        print("EVOKE_EVAL_MODE_R5", json.dumps({
            "training_before_correction": before,
            "training_after_correction": bool(transformer.training),
            "transformer_signature": id(transformer),
            "formal_output_accepted": False}), flush=True)
        return result
    engine.build_pipe = build
    return engine
