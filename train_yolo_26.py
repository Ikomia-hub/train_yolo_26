"""
Main Ikomia plugin module.
Ikomia Studio and Ikomia API use it to load algorithms dynamically.
"""
from ikomia import dataprocess
from train_yolo_26.train_yolo_26_process import TrainYolo26Factory
from train_yolo_26.train_yolo_26_process import TrainYolo26ParamFactory


class IkomiaPlugin(dataprocess.CPluginProcessInterface):
    """
    Interface class to integrate the process with Ikomia application.
    Inherits PyDataProcess.CPluginProcessInterface from Ikomia API.
    """
    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        """Instantiate process object."""
        return TrainYolo26Factory()

    def get_widget_factory(self):
        """Instantiate associated widget object."""
        from train_yolo_26.train_yolo_26_widget import TrainYolo26WidgetFactory
        return TrainYolo26WidgetFactory()

    def get_param_factory(self):
        """Instantiate algorithm parameters object."""
        return TrainYolo26ParamFactory()
