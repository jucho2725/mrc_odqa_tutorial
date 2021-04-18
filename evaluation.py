import  json

def evaluation(gt_path, pred_path):
    """Calculate EM score.

    Input file format:
        Each line has answer for each problem.
        e.g. {"text": "프랑스"}
    """
    correct = 0.
    total = 0.

    gt = open(gt_path, 'r').readlines()
    pred = open(pred_path, 'r').readlines()

    assert len(gt) == len(pred), f'Lengths of gt and pred must be same Len(gt):{len(gt)} != Len(pred):{len(pred)}'

    for g, p in zip(gt, pred):
        p_data = json.loads(p)
        g_data = json.loads(g)

        if p_data['text'] == g_data['text']:
            correct +=  1
        total += 1

    result = f'{round(correct * 100 / total, 2)}%'

    return result
