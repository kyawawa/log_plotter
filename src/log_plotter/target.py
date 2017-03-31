from abc import ABCMeta, abstractmethod
import numpy

class ClassProperty(property):
    '''
    utility class to make property from classmethod
    '''
    def __get__(self, cls, owner):
        return self.fget.__get__(None, owner)()

class TargetFamilyDict(dict):
    '''
    Dictionary for targets belonging to same log path.
    '''
    all_target_class = set()
    def __init__(self):
        dict.__init__(self)
        self.path = '' # path/to/log
        for target_class in TargetFamilyDict.all_target_class:
            self[target_class.name] = target_class(target_dict=self)

    def __getitem__(self, name):
        try:
            return dict.__getitem__(self, name)
        except KeyError:
            return self.add_logged_target(name)

    def add_target(self, target_class):
        TargetFamilyDict.all_target_class.add(target_class)
        trgt = target_class()
        self[target_class.name] = trgt
        return trgt

    def add_logged_target(self, name):
        cls = type(name, (LoggedTarget,),{})
        cls.depends.add(name)
        return self.add_target(cls)

    def add_timed_target(self, name=None, depends=None, func=None):
        if name is None or func is None:
            return
        cls = type(name, (TimedTarget,), {"func": staticmethod(func)})
        if depends is not None:
            try: # depends is iterable
                cls.depends.update(depends)
            except TypeError: # depends is str
                cls.depends.add(depends)
        return self.add_target(cls)


class TargetInterface(object):
    __metaclass__ = ABCMeta
    depends = set()

    def __init__(self, target_dict={}):
        self.index_to_use = None
        self.target_dict = target_dict
        self.index_to_use = []

    @ClassProperty
    @classmethod
    def name(cls):
        return cls.__name__

    @abstractmethod
    def get_x_data(self, index, *args, **kwargs):
        pass

    @abstractmethod
    def get_y_data(self, index, *args, **kwargs):
        pass

class LoggedTarget(TargetInterface):
    def __init__(self, target_dict={}):
        TargetInterface.__init__(self, target_dict={})
        self.data = None

    def get_x_data(self, index):
        return self.data[:, 0]

    def get_y_data(self, index):
        return self.data[:, 1+index]

class TimedTarget(TargetInterface):
    '''
    This is abstract class to generate target whose x is time data

    TimedTarget.func(index, target_dict, *args, **kwargs) -> one dimensional numpy.ndarray
    '''
    __metaclass__ = ABCMeta
    def __init__(self, target_dict={}, args=[], kwargs={}):
        TargetInterface.__init__(self, target_dict={})
        self.args = args
        self.kwargs = kwargs

    def get_x_data(self, index):
        return list(self.depends)[0].get_x_data(0)

    def get_y_data(self, index):
        return self.func(index, self.target_dict, *self.args, **self.kwargs)

    @staticmethod
    def func(index, target_dict, *args, **kwargs):
        '''
        to be overwritten
        '''
        pass

    @staticmethod
    def from_func(name=None, depends=None):
        def func(f):
            if name is None:
                raise TypeError, "name is None"
            cls = type(name, (TimedTarget,), {"func": staticmethod(f)})
            if depends is not None:
                try: # depends is iterable
                    cls.depends.update(depends)
                except TypeError: # depends is str
                    cls.depends.add(depends)
            TargetFamilyDict.all_target_class.add(cls)
            return f
        return func
