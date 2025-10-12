def flatten_with_paths(data, path=()):
    """将三维嵌套结构展平成 (path, value) 的列表"""
    if not isinstance(data, list):
        return [(path, data)]
    res = []
    for i, x in enumerate(data):
        res.extend(flatten_with_paths(x, path + (i,)))
    return res


def reconstruct_from_paths(paths, values):
    """根据路径信息将 values 还原成原始嵌套结构"""
    from copy import deepcopy
    # 找出最大深度的索引结构
    res = {}
    for (p, v) in zip(paths, values):
        d = res
        for i in p[:-1]:
            d = d.setdefault(i, {})
        d[p[-1]] = v

    # 递归地将 dict 转回 list
    def dict_to_list(d):
        if not isinstance(d, dict):
            return d
        max_idx = max(d.keys())
        return [dict_to_list(d[i]) for i in range(max_idx + 1)]

    return dict_to_list(res)


# ==== 示例 ====

data = [
    1, 2, 3
]

# 1️⃣ 拍平数据
flat_with_paths = flatten_with_paths(data)
paths, flat = zip(*flat_with_paths)

# 2️⃣ 调用处理函数
def process(flat):
    return [[x, x * 10] for x in flat]

processed = process(flat)

# 3️⃣ 还原结构
restored = reconstruct_from_paths(paths, processed)
import json
print(json.dumps(restored, indent=2))
