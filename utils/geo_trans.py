from shapely.geometry import MultiPolygon, Polygon
from shapely.affinity import translate, scale, rotate


def boolean_operation(profile1: Polygon, profile2: Polygon, operation="union"):
    """几何轮廓布尔运算"""
    if not (
        isinstance(profile1, (Polygon, MultiPolygon))
        and isinstance(profile2, (Polygon, MultiPolygon))
    ):
        raise TypeError("输入轮廓类型必须是 Polygon 或 MultiPolygon")
    if operation == "union":
        return profile1.union(profile2)
    elif operation == "difference":
        return profile1.difference(profile2)
    elif operation == "intersection":
        return profile1.intersection(profile2)
    else:
        raise ValueError("不支持的布尔运算类型")


def transform_operation(
    profile: Polygon,
    transform="translate",
    translate_param=(0, 0),
    scale_param=(1, (0, 0)),
    rotate_param=(0, (0, 0)),
):
    """几何轮廓变换操作"""
    if not isinstance(profile, (Polygon, MultiPolygon)):
        raise TypeError("输入轮廓类型必须是 Polygon 或 MultiPolygon")
    # 平移
    if transform == "translate":
        return translate(
            profile,
            xoff=translate_param[0],
            yoff=translate_param[1],
        )
    # 缩放
    elif transform == "scale":
        profile = scale(
            profile,
            xfact=scale_param[0],
            yfact=scale_param[0],
            origin=scale_param[1],
        )
    # 旋转，逆时针为负，顺时针为正
    elif transform == "rotate":
        profile = rotate(
            profile,
            rotate_param[0],
            origin=rotate_param[1],
            use_radians=True,
        )
    else:
        raise ValueError("不支持的变换类型")

    return profile
