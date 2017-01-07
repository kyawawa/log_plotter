#!/usr/bin/env python
import numpy
import struct
import math
import sys
from scipy import signal
from IPython.core.debugger import Tracer

try:
    import pyqtgraph
except:
    print "please install pyqtgraph. see http://www.pyqtgraph.org/"
    sys.exit(1)

def rotationMatrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = numpy.asarray(axis)
    axis = axis/math.sqrt(numpy.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return numpy.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def butterLowpassFilter(data, cutoff, fs = 500, order = 2):
    normal_cutoff = cutoff / (0.5 * fs)
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    # b, a = signal.bessel(order, normal_cutoff, btype='low', analog=False)
    return signal.lfilter(b, a, data)

class PlotMethod(object):
    urata_len = 16
    # color_list = pyqtgraph.functions.Colors.keys()
    # default color set on gnuplot 5.0
    color_list = ["9400D3", "009E73", "56B4E9", "E69F00", "F0E442", "0072B2", "E51E10", "0000FF"]
    linetypes = {
        "color": color_list * 2,
        "style": [pyqtgraph.QtCore.Qt.SolidLine] * len(color_list) + [pyqtgraph.QtCore.Qt.DotLine] * len(color_list)
    }


    @staticmethod
    def __plot_urata_servo(plot_item, times, data_dict, logs, log_cols, cur_col, key, i, offset1, offset2=1):
        plot_item.plot(times, data_dict[logs[0]][:, (PlotMethod.urata_len+1) * log_cols[0] + (offset1+offset2)],
                       pen=pyqtgraph.mkPen(PlotMethod.linetypes["color"][i], width=2, style=PlotMethod.linetypes["style"][i]), name=key)

    @staticmethod
    def plot_servostate(plot_item, times, data_dict, logs, log_cols, cur_col, key, i):
        def RePack(x):
            val = struct.unpack('i', struct.pack('f', float(x)))[0]
            #calib = (val & 0x01)
            #servo = (val & 0x02) >> 1
            #power = (val & 0x04) >> 2
            state = (val & 0x0007fff8) >> 3
            #temp  = (val & 0xff000000) >> 24
            return state
        vfr = numpy.vectorize(RePack)
        plot_item.plot(times, vfr(data_dict[logs[0]][:, (PlotMethod.urata_len+1) * log_cols[0] + (0+0)]),
                       pen=pyqtgraph.mkPen('r', width=2, style=PlotMethod.linetypes["style"][i]), name=key)

    @staticmethod
    def plot_commnormal(plot_item, times, data_dict, logs, log_cols, cur_col, key, i):
        PlotMethod.__plot_urata_servo(plot_item, times, data_dict, logs, log_cols, cur_col, key, i, 13)

    @staticmethod
    def plot_12V(plot_item, times, data_dict, logs, log_cols, cur_col, key, i):
        PlotMethod.__plot_urata_servo(plot_item, times, data_dict, logs, log_cols, cur_col, key, i, 9)

    @staticmethod
    def plot_80V(plot_item, times, data_dict, logs, log_cols, cur_col, key, i):
        PlotMethod.__plot_urata_servo(plot_item, times, data_dict, logs, log_cols, cur_col, key, i, 2)

    @staticmethod
    def plot_current(plot_item, times, data_dict, logs, log_cols, cur_col, key, i):
        PlotMethod.__plot_urata_servo(plot_item, times, data_dict, logs, log_cols, cur_col, key, i, 1)

    @staticmethod
    def plot_motor_temp(plot_item, times, data_dict, logs, log_cols, cur_col, key, i):
        PlotMethod.__plot_urata_servo(plot_item, times, data_dict, logs, log_cols, cur_col, key, i, 0)

    @staticmethod
    def plot_motor_outer_temp(plot_item, times, data_dict, logs, log_cols, cur_col, key, i):
        PlotMethod.__plot_urata_servo(plot_item, times, data_dict, logs, log_cols, cur_col, key, i, 7)

    @staticmethod
    def plot_pgain(plot_item, times, data_dict, logs, log_cols, cur_col, key, i):
        PlotMethod.__plot_urata_servo(plot_item, times, data_dict, logs, log_cols, cur_col, key, i, 10)

    @staticmethod
    def plot_dgain(plot_item, times, data_dict, logs, log_cols, cur_col, key, i):
        PlotMethod.__plot_urata_servo(plot_item, times, data_dict, logs, log_cols, cur_col, key, i, 11)

    @staticmethod
    def plot_abs_enc(plot_item, times, data_dict, logs, log_cols, cur_col, key, i):
        plot_item.plot(times, [math.degrees(x) for x in data_dict[logs[0]][:, (PlotMethod.urata_len+1) * log_cols[0] + (6+1)]],
                       pen=pyqtgraph.mkPen('g', width=2, style=PlotMethod.linetypes["style"][i]), name=key)

    @staticmethod
    def plot_rh_q_st_q(plot_item, times, data_dict, logs, log_cols, cur_col, key, i):
        plot_item.plot(times, [math.degrees(x) for x in (data_dict[logs[1]][:, log_cols[1]] - data_dict[logs[0]][:, log_cols[0]])],
                       pen=pyqtgraph.mkPen('r', width=2, style=PlotMethod.linetypes["style"][i]), name=key)

    @staticmethod
    def plot_rad2deg(plot_item, times, data_dict, logs, log_cols, cur_col, key, i):
        data_rad=data_dict[logs[0]][:, log_cols[0]]
        data_deg=[math.degrees(x) for x in data_rad]
        plot_item.plot(times, data_deg,pen=pyqtgraph.mkPen(PlotMethod.linetypes["color"][i], width=(len(logs)-i+3), style=PlotMethod.linetypes["style"][i]), name=key)

    @staticmethod
    def plot_watt(plot_item, times, data_dict, logs, log_cols, cur_col, key, i):
        joint_vel=data_dict[logs[0]][:, log_cols[0]]
        joint_tau=data_dict[logs[1]][:, log_cols[1]]
        watt=joint_vel*joint_tau
        plot_item.plot(times, watt,pen=pyqtgraph.mkPen(PlotMethod.linetypes["color"][i], width=len(logs)-i, style=PlotMethod.linetypes["style"][i]), name=key, fillLevel=0, fillBrush=PlotMethod.linetypes["color"][i])

    @staticmethod
    def plot_sum(plot_item, times, data_dict, logs, log_cols, cur_col, key, i):
        data = numpy.sum([data_dict[logs[j]][:, log_cols[j]] for j in range(len(log_cols))], axis = 0)
        plot_item.plot(times, data, pen=pyqtgraph.mkPen(PlotMethod.linetypes["color"][i], width=2, style=PlotMethod.linetypes["style"][i]), name=key)

    @staticmethod
    def plot_diff(plot_item, times, data_dict, logs, log_cols, cur_col, key, i):
        data_minuend = data_dict[logs[0]][:, log_cols[0]]
        data_subtrahend = data_dict[logs[1]][:, log_cols[1]]
        data = data_minuend - data_subtrahend
        plot_item.plot(times, data, pen=pyqtgraph.mkPen(PlotMethod.linetypes["color"][i], width=len(logs)-i, style=PlotMethod.linetypes["style"][i]), name=key)

    @staticmethod
    def plot_rad2deg_diff(plot_item, times, data_dict, logs, log_cols, cur_col, key, i):
        plot_item.plot(times, [math.degrees(x) for x in (data_dict[logs[1]][:, log_cols[1]] - data_dict[logs[0]][:, log_cols[0]])],
                       pen=pyqtgraph.mkPen('r', width=2, style=PlotMethod.linetypes["style"][i]), name=key)

    @staticmethod
    def plot_comp(plot_item, times, data_dict, logs, log_cols, cur_col, key, i):
        plot_item.plot(times, data_dict[logs[0]][:, log_cols[0]],
                       pen=pyqtgraph.mkPen(PlotMethod.linetypes["color"][i], width=1+len(logs)-i, style=PlotMethod.linetypes["style"][i]), name=key)
        if log_cols[0] % 6 < 3: # position
            plot_item.setYRange(-0.025, +0.025) # compensation limit
        else: # rotation
            plot_item.setYRange(math.radians(-10), math.radians(+10)) # compensation limit

    @staticmethod
    def plot_COP(plot_item, times, data_dict, logs, log_cols, cur_col, key, i):
        offset = log_cols[0]*6
        arg = logs[min(len(logs)-1,cur_col)]
        f_z = data_dict[arg][:, offset+2]
        tau_x = data_dict[arg][:, offset+3]
        tau_y = data_dict[arg][:, offset+4]
        plot_item.plot(times, -tau_y/f_z, pen=pyqtgraph.mkPen(PlotMethod.color_list[2*i], width=2, style=PlotMethod.linetypes["style"][i]), name=key)
        plot_item.plot(times,  tau_x/f_z, pen=pyqtgraph.mkPen(PlotMethod.color_list[2*i+1], width=2, style=PlotMethod.linetypes["style"][i]), name=key)

    @staticmethod
    def plot_inverse(plot_item, times, data_dict, logs, log_cols, cur_col, key, i):
        plot_item.plot(times, -data_dict[logs[0]][:, log_cols[0]], pen=pyqtgraph.mkPen(PlotMethod.linetypes["color"][i], width=2, style=PlotMethod.linetypes["style"][i]), name=key)

    @staticmethod
    def plot_swing_wrench(plot_item, times, data_dict, logs, log_cols, cur_col, key, i):
        wrench = data_dict[logs[0]][:, log_cols[0]]
        contact_states = data_dict[logs[1]][:, log_cols[1]]
        plot_item.plot(times, [x * (1 - y) for x, y in zip(wrench, contact_states)], pen=pyqtgraph.mkPen(PlotMethod.linetypes["color"][i], width=2, style=PlotMethod.linetypes["style"][i]), name=key)

    @staticmethod
    def plot_swing_wrench_compensation_filter(plot_item, times, data_dict, logs, log_cols, cur_col, key, i):
        wrench = data_dict[logs[0]][:, log_cols[0]]
        contact_states = data_dict[logs[1]][:, log_cols[1]]
        ee_pos_acc = data_dict[logs[2]][:, log_cols[2]]
        ee_pos_acc = butterLowpassFilter(ee_pos_acc, 20.0, 500.0, 2)

        foot_mass = 0.665
        wrench += foot_mass * ee_pos_acc
        plot_item.plot(times, [x * (1 - y) for x, y in zip(wrench, contact_states)], pen=pyqtgraph.mkPen(PlotMethod.linetypes["color"][i], width=2, style=PlotMethod.linetypes["style"][i]), name=key)
        # plot_item.plot(times, ee_pos_acc, pen=pyqtgraph.mkPen(PlotMethod.linetypes["color"][i], width=2, style=PlotMethod.linetypes["style"][i]), name=key)

    @staticmethod
    def plot_data_lowpass_filtered(plot_item, times, data_dict, logs, log_cols, cur_col, key, i):
        data = data_dict[logs[0]][:, log_cols[0]]
        data = butterLowpassFilter(data, 20.0, 500.0, 2)
        plot_item.plot(times, data, pen=pyqtgraph.mkPen(PlotMethod.linetypes["color"][i], width=2, style=PlotMethod.linetypes["style"][i]), name=key)


    @staticmethod
    def plot_swing_wrench_compensation_from_pos(plot_item, times, data_dict, logs, log_cols, cur_col, key, i):
        # wrench = numpy.array([sensorR.dot(x) for x in data_dict[logs[0]][:, 3:]])
        wrench = data_dict[logs[0]][:, log_cols[0]]
        # wrench = wrench[:, log_cols[0]]
        contact_states = data_dict[logs[1]][:, log_cols[1]]
        ee_pos = data_dict[logs[2]][:, log_cols[2]]
        ee_pos = butterLowpassFilter(ee_pos, 2.0, 500.0, 2)
        dt = times[1] - times[0]
        ee_pos_vel = numpy.diff(ee_pos) / dt
        ee_pos_vel = numpy.append([0], ee_pos_vel)
        ee_pos_acc = numpy.diff(ee_pos_vel) / dt
        ee_pos_acc = numpy.append([0], ee_pos_acc)
        foot_mass = 0.665
        foot_mass = 1.8
        wrench += foot_mass * ee_pos_acc
        plot_item.plot(times, [x * (1 - y) for x, y in zip(wrench, contact_states)], pen=pyqtgraph.mkPen(PlotMethod.linetypes["color"][i], width=2, style=PlotMethod.linetypes["style"][i]), name=key)
        # plot_item.plot(times, ee_pos_acc, pen=pyqtgraph.mkPen(PlotMethod.linetypes["color"][i], width=2, style=PlotMethod.linetypes["style"][i]), name=key)

    @staticmethod
    def plot_acc_from_filtered_pos(plot_item, times, data_dict, logs, log_cols, cur_col, key, i):
        ee_pos = data_dict[logs[0]][:, log_cols[0]]
        ee_pos = butterLowpassFilter(ee_pos, 2.0, 500.0, 2)
        dt = times[1] - times[0]
        ee_pos_vel = numpy.diff(ee_pos) / dt
        ee_pos_vel = numpy.append([0], ee_pos_vel)
        ee_pos_acc = numpy.diff(ee_pos_vel) / dt
        ee_pos_acc = numpy.append([0], ee_pos_acc)
        plot_item.plot(times, ee_pos_acc, pen=pyqtgraph.mkPen(PlotMethod.linetypes["color"][i], width=2, style=PlotMethod.linetypes["style"][i]), name=key)

    @staticmethod
    def plot_st_actcontactstates(plot_item, times, data_dict, logs, log_cols, cur_col, key, i):
        plot_item.plot(times, numpy.add(data_dict[logs[0]][:, log_cols[0]], 0.8) / 2.0, pen=pyqtgraph.mkPen(PlotMethod.linetypes["color"][i], width=2, style=PlotMethod.linetypes["style"][i]), name=key)

    @staticmethod
    def normal(plot_item, times, data_dict, logs, log_cols, cur_col, key, i):
        plot_item.plot(times, data_dict[logs[0]][:, log_cols[0]], pen=pyqtgraph.mkPen(PlotMethod.linetypes["color"][i], width=2, style=PlotMethod.linetypes["style"][i]), name=key)
