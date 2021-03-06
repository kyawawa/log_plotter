#!/usr/bin/env python

import argparse
import csv
import functools
import math
import multiprocessing
import numpy
import struct
import sys
import time
import yaml
import log_plotter.plot_method as plot_method
from log_plotter.graph_legend import GraphLegendInfo, expand_str_to_list

try:
    import pyqtgraph
except:
    print "please install pyqtgraph. see http://www.pyqtgraph.org/"
    sys.exit(1)


# decorator for time measurement
def my_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        print("start : " + func.func_name)
        result = func(*args, **kwargs)
        print('{0:>10} : {1:3.3f} [s]'.format(func.func_name,
                                              time.time() - start))
        return result
    return wrapper


# seems that we should declare global function for multiprocess
def readOneTopic(fname):
    tmp = []
    try:
        with open(fname, 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                dl = filter(lambda x: x != '', row)
                tmp.append([float(x) for x in dl])
    except Exception as e:
        print '[readOneToopic] error occured while reading {}'.format(fname)
        raise e
    return numpy.array(tmp)


class DataloggerLogParser:
    def __init__(self, fname, plot_conf_name, layout_conf_name, title):
        self.fname = fname
        with open(plot_conf_name, "r") as f:
            self.plot_dict = yaml.load(f)
        with open(layout_conf_name, "r") as f:
            self.layout_list = yaml.load(f)
        # expand [0-33] => [0,1,2,...,33]
        for leg_info in self.plot_dict.values():
            for log_info in leg_info['data']:
                if type(log_info['column'][0]) == str:
                    log_info['column'] = expand_str_to_list(log_info['column'][0])
        for group in self.layout_list:
            for leg in group['legends']:
                if type(leg['id'][0]) == str:
                    leg['id'] = expand_str_to_list(leg['id'][0])
            group.setdefault('newline', True)

        # setup view
        self.view = pyqtgraph.GraphicsLayoutWidget()
        self.view.setBackground('w')
        self.view.setWindowTitle(title if title else fname.split('/')[-1])
        # self.dateListDict is set by self.readData()
        self.dataListDict = {}# todo: to list of dictionary
        # back up for plot items
        self.plotItemOrig = {}

    @my_time
    def readData(self):
        '''
        read log data from log files and store dataListDict
        # self.dataListDict[topic] = numpy.array([[t_0, x_0, y_0, ...],
        #                                         [t_1, x_1, y_1, ...],
        #                                         ...,
        #                                         [t_n, x_n, y_n, ...]])
        '''
        # get list fo used topic
        all_legends = reduce(lambda x,y: x+y, [group['legends'] for group in self.layout_list])
        used_keys = list(set([legend['key'] for legend in all_legends]))
        log_col_pairs = reduce(lambda x,y: x+y, [self.plot_dict[key]['data'] for key in used_keys])
        topic_list = list(set([log_col_pair['log'] for log_col_pair in log_col_pairs]))
        self._used_keys = used_keys
        self._topic_list = topic_list

        # store data in parallel
        fname_list = [self.fname+'.'+ext for ext in topic_list]
        pl = multiprocessing.Pool()
        data_list = pl.map(readOneTopic, fname_list)
        for topic, data in zip(topic_list, data_list):
            self.dataListDict[topic] = data
        # set the fastest time as 0
        min_time = min([self.dataListDict[topic][0][0] for topic in topic_list])
        for topic in topic_list:
            raw_time = self.dataListDict[topic][:, 0]
            self.dataListDict[topic][:, 0] = [x - min_time for x in raw_time]
        # fix servoState
        if 'RobotHardware0_servoState' in topic_list:
            def servoStatesConverter(x):
                return struct.unpack('f', struct.pack('i', int(x)))[0]
            vf = numpy.vectorize(servoStatesConverter)
            ss_tmp = self.dataListDict['RobotHardware0_servoState'][:, 1:]
            self.dataListDict['RobotHardware0_servoState'][:, 1:] = vf(ss_tmp)

    @my_time
    def setLayout(self):
        '''
        set layout of view according to self.plot_dict
        '''
        # set view and get legend info
        self.legend_list = [[]]
        graph_row = 0
        graph_col = 0
        for i, group in enumerate(self.layout_list):
            group_len = max(len(leg['id']) for leg in group['legends'])
            for j in range(group_len):
                # add graph
                plot_item = self.view.addPlot()
                self.legend_list[graph_row].append([])
                plot_item.setTitle(group['title']+" "+str(j))
                plot_item.showGrid(x=True, y=True)
                if group.has_key('downsampling'):
                    plot_item.setDownsampling(ds = group['downsampling'].get('ds', 100),
                                              auto=group['downsampling'].get('auto', False),
                                              mode=group['downsampling'].get('mode', 'peak'))

                # add legend info to this graph
                for k in range(len(group['legends'])):
                    legend_info = GraphLegendInfo(self.layout_list, self.plot_dict, i, j, k)
                    self.legend_list[graph_row][graph_col].append(legend_info)
                graph_col += 1
            if group['newline']:
                # add newline
                self.view.nextRow()
                graph_row +=1
                graph_col = 0
                self.legend_list.append([])

    @my_time
    def plotData(self):
        '''
        plot
        '''

        color_list = pyqtgraph.functions.Colors.keys()
        times = self.dataListDict[self._topic_list[0]][:, 0]
        data_dict = {}
        for log in self._topic_list: data_dict[log] = self.dataListDict[log][:, 1:]
        # self.legend_list = [[[leg1, leg2,...],[],...]
        #                     [[],              [],...]
        #                     [[],              [],...]]
        for i, group_legends in enumerate(self.legend_list):
            for j, graph_legends in enumerate(group_legends):
                cur_item = self.view.ci.rows[i][j]
                cur_item.addLegend(offset=(0, 0))
                for k, legend in enumerate(graph_legends):
                    func = legend.info['func']
                    logs = [d['log'] for d in legend.info['data']]
                    log_cols = [d['column'] for d in legend.info['data']]
                    cur_col = j
                    key = legend.info['label']
                    getattr(plot_method.PlotMethod, func)(cur_item, times, data_dict, logs, log_cols, cur_col, key, k)

    @my_time
    def setLabel(self):
        '''
        set label: time for bottom plots, unit for left plots
        '''
        row_num = len(self.view.ci.rows)
        # left plot items
        for i in range(row_num):
            cur_item = self.view.ci.rows[i][0]
            title = cur_item.titleLabel.text
            tmp_units = None
            if ("12V" in title) or ("80V" in title):
                tmp_units = "V"
            elif "current" in title:
                tmp_units = "A"
            elif ("temperature" in title) or ("joint_angle" in title) or ("attitude" in title) or ("tracking" in title):
                tmp_units = "deg"
            elif ("joint_velocity" in title):
                tmp_units = "deg/s"
            elif ("watt" in title):
                tmp_units = "W"
            cur_item.setLabel("left", text="", units=tmp_units)
            # we need this to suppress si-prefix until https://github.com/pyqtgraph/pyqtgraph/pull/293 is merged
            for ax in cur_item.axes.values():
                ax['item'].enableAutoSIPrefix(enable=False)
                ax['item'].autoSIPrefixScale = 1.0
                ax['item'].labelUnitPrefix = ''
                ax['item'].setLabel()
        # bottom plot items
        col_num = len(self.view.ci.rows[row_num-1])
        for i in range(col_num):
            cur_item = self.view.ci.rows[row_num-1][i]
            cur_item.setLabel("bottom", text="time", units="s")

    @my_time
    def linkAxes(self):
        '''
        link all X axes and some Y axes
        '''
        # X axis
        all_items = self.view.ci.items.keys()
        target_item = all_items[0]
        for i, p in enumerate(all_items):
            if i != 0:
                p.setXLink(target_item)
            else:
                p.enableAutoRange()
        # Y axis
        for cur_row_dict in self.view.ci.rows.values():
            all_items = cur_row_dict.values()
            target_item = all_items[0]
            title = target_item.titleLabel.text
            if title.find("joint_angle") == -1 and title.find("_force") == -1 and title != "imu" and title.find("comp") == -1:
                y_min = min([ci.viewRange()[1][0] for ci in all_items])
                y_max = max([ci.viewRange()[1][1] for ci in all_items])
                target_item.setYRange(y_min, y_max)
                for i, p in enumerate(all_items):
                    if i != 0:
                        p.setYLink(target_item)

    @my_time
    def customMenu(self):
        '''
        customize right-click context menu
        '''
        self.plotItemOrig = self.view.ci.items.copy()
        all_items = self.view.ci.items.keys()
        for pi in all_items:
            vb = pi.getViewBox()
            hm = vb.menu.addMenu('Hide')
            qa1 = hm.addAction('hide this plot')
            qa2 = hm.addAction('hide this row')
            qa3 = hm.addAction('hide this column')
            qa4 = vb.menu.addAction('restore plots')
            qa5 = hm.addAction('hide except this plot')
            qa6 = hm.addAction('hide except this row')
            qa7 = hm.addAction('hide except this colmn')
            def hideCB(item):
                self.view.ci.removeItem(item)
            def hideRowCB(item):
                r, _c = self.view.ci.items[item][0]
                del_list = [self.view.ci.rows[r][c] for c in self.view.ci.rows[r].keys()]
                for i in del_list:
                    self.view.ci.removeItem(i)
            def hideColCB(item):
                _r, c = self.view.ci.items[item][0]
                del_list = []
                row_num = len(self.view.ci.rows)
                for r in range(row_num):
                    if c in self.view.ci.rows[r].keys():
                        del_list.append(self.view.ci.rows[r][c])
                for i in del_list:
                    self.view.ci.removeItem(i)
            def hideExcCB(item):
                del_list = self.view.ci.items.keys()
                del_list.remove(item)
                for i in del_list:
                    self.view.ci.removeItem(i)
            def hideExcRowCB(item):
                del_list = self.view.ci.items.keys()
                r, _c = self.view.ci.items[item][0]
                not_del_list=[self.view.ci.rows[r][c] for c in self.view.ci.rows[r].keys()]
                for i in del_list:
                    if i in not_del_list:
                        del_list.remove(i)
                for i in del_list:
                    self.view.ci.removeItem(i)
            def hideExcColumnCB(item):
                del_list = self.view.ci.items.keys()
                _r, c = self.view.ci.items[item][0]
                not_del_list=[self.view.ci.rows[r][c] for r in range(len(a.view.ci.rows))]
                for i in del_list:
                    if i in not_del_list:
                        del_list.remove(i)
                for i in del_list:
                    self.view.ci.removeItem(i)
            def restoreCB():
                self.view.ci.clear()
                for key in self.plotItemOrig:
                    r, c = self.plotItemOrig[key][0]
                    self.view.ci.addItem(key, row=r, col=c)
            qa1.triggered.connect(functools.partial(hideCB, pi))
            qa2.triggered.connect(functools.partial(hideRowCB, pi))
            qa3.triggered.connect(functools.partial(hideColCB, pi))
            qa4.triggered.connect(restoreCB)
            qa5.triggered.connect(functools.partial(hideExcCB, pi))
            qa6.triggered.connect(functools.partial(hideExcRowCB, pi))
            qa7.triggered.connect(functools.partial(hideExcColumnCB, pi))

    def main(self):
        '''
        1. read log files
        2. decide layout
        3. plot data
        4. set label
        5. link axes
        6. customize context menu
        7. show
        '''
        self.readData()
        self.setLayout()
        self.plotData()
        self.setLabel()
        self.linkAxes()
        self.customMenu()
        self.view.showMaximized()

def main():
    # args
    parser = argparse.ArgumentParser(description='plot data from hrpsys log')
    parser.add_argument('-f', type=str, help='input file', metavar='file', required=True)
    parser.add_argument('--plot', type=str, help='plot configure file', metavar='file', required=True)
    parser.add_argument('--layout', type=str, help='layout configure file', metavar='file', required=True)
    parser.add_argument('-t', type=str, help='title', default=None)
    parser.add_argument("-i", action='store_true', help='interactive (start IPython)')
    parser.set_defaults(feature=False)
    args = parser.parse_args()
    # main
    app = pyqtgraph.Qt.QtGui.QApplication([])
    a = DataloggerLogParser(args.f, args.plot, args.layout, args.t)
    a.main()

    if args.i:
        [app.processEvents() for i in range(2)]
        # start ipython
        print '====='
        print "please use \033[33mapp.processEvents()\033[m to update graph."
        print "you can use \033[33ma\033[m as DataloggerLogParser instance."
        print '====='
        from IPython import embed
        embed()
    else:
        pyqtgraph.Qt.QtGui.QApplication.instance().exec_()

if __name__ == '__main__':
    main()

