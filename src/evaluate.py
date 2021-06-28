from src.metrics import recall_at_k


def evaluate(ease_model, eval_tr, eval_te, k):
    predictions = ease_model.predict(eval_tr, k)

    uid_to_prediction = dict(zip(eval_tr.uid.unique(), predictions.numpy()))

    preds_with_true = eval_te.groupby("uid").agg({'sid': list})
    preds_with_true = preds_with_true.reset_index()
    preds_with_true['preds'] = preds_with_true.apply(lambda x: uid_to_prediction.get(x.uid), axis=1)

    return preds_with_true.apply(lambda x: recall_at_k(x.sid, x.preds, k), axis=1).mean()
