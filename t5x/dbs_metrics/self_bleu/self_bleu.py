import sacrebleu

def self_bleu(targets, predictions):
    """
    Computes Self-BLEU over a list of lists of predictions.
    
    Args:
      targets: list of strings (ignored).
      predictions: list of lists of strings (each list contains N decodes for a single input).
    """
    total_score = 0.0
    valid_count = 0
    
    for preds in predictions:
        # predictions can be multiple decodes per input
        if len(preds) <= 1:
            continue
        
        cand_score = 0.0
        for i, hyp in enumerate(preds):
            refs = [preds[j] for j in range(len(preds)) if i != j]
            # SacreBLEU sentence_bleu expects a hypothesis string and a list of reference strings
            bleu = sacrebleu.sentence_bleu(hyp, refs).score
            cand_score += bleu
            
        total_score += cand_score / len(preds)
        valid_count += 1
        
    return {"self_bleu": total_score / valid_count if valid_count > 0 else 0.0}