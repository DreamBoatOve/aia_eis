"""
records where this IS comes from?
    experiment
        files in DTA ... formats
    paper
        figures
    simulation
        software
            ZSimpWin
            ZView
            AIA-EIS
"""
class IS_scource:
    # def __init__(self, source_type:str):
    def __init__(self):
        """
        :param
            source_type: str
                'experiment'
                'paper'
                'simulation'
                    software
                        Commercial
                            ZSimpWin
                            ZView
                        Open Source
                            Impedance
                            PyEIS
                            AIA-EIS
                'web'
        """

        pass

    def form4Experiment(self):
        """
        who
        :return:
        """
        pass
    def form4Paper(self):
        """
        DOI
        Fig number
            line number
        :return:
        """
        pass
    def form4Simulation(self):
        """
        software
        :return:
        """
        pass
    def fill4Simulation(self, *args, **kwargs):
        """
        和网页中填写一样的信息，这个是在代码里一样一样的填
        :return:
        """
        commercial_bool = kwargs['commercial']
        software_name = kwargs['software']
        pass

    def fill4Experiment(self, *args, **kwargs):
        """
        和网页中填写一样的信息，这个是在代码里一样一样的填
        :return:
        """
        if 'peopleName' in kwargs.keys():
            peopleName = kwargs['peopleName']
        # date
        # location
        # ....
        pass