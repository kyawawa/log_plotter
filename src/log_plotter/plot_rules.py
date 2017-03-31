#!/usr/bin/env python
import numpy

# ログをプロットするだけのtarget
class st_q(LoggedTarget): pass
class abc_q(LoggedTarget): pass
# 横軸は時刻で，縦軸はログどうしの演算でプロット
class watt(TimedTarget):
    def get_y_data(self):
        return None
class joint(TimedTarget):
    def get_y_data(self):
        return None
class joint_diff(TimedTarget):
    def get_y_data(self):
        return None
# 複雑な演算
class sin_curve(TargetInterface): pass
class coords2coords(TargetInterface): pass
