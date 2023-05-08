import logging
from abc import ABCMeta, abstractmethod
import pandas as pd

from job_queue import PriorityJobQueue
from utils.average_meter import AverageMeter


class Scheduler(metaclass=ABCMeta):

    def __init__(self,
                contention_aware: bool = False,
                round_duration=300,
                isolated_thp_path='./jobs/6-workload-isolated-thp.csv',
                shared_thp_path='./jobs/6-workload-shared-thp.csv',
                contention_sensitivity_path='./jobs/6-workload-contention-sensitivity.csv',
                contention_sensitivity_knob: float = 0.2):

        self.job_queue = PriorityJobQueue()
        self.contention_aware = contention_aware
        self.round_duration = round_duration
        self._logger = logging.getLogger('scheduler')

        self.system_thp = AverageMeter()
        self.makespan = AverageMeter()

        # Read isolated thp and convert to dataframe
        self.__parse_isolated_thp(isolated_thp_path)

        # Read shared_thp and convert to dataframe
        self.__parse_shared_thp(shared_thp_path)

        # Read contention_sensitivity and convert to dataframe
        if contention_aware:
            self.contention_sensitivity_knob = contention_sensitivity_knob
            self.__parse_contention_sensitivity(contention_sensitivity_path)

        # self._logger.info(self)


    def parse_json_job(self, json):
        pass


    def submit(self, json_job):
        self.job_queue.put(self.parse_json_job(json_job))


    def isolated_thp_get(self, row, col='isolated thp'):
        return self.isolated_thp.at[row, col]


    def shared_thp_get(self, row, col):
        return self.shared_thp.at[row, col]


    def contention_sensitivity_get(self, row, col):
        assert self.contention_sensitivity is not None
        return self.contention_sensitivity.at[row, col]


    def __parse_isolated_thp(self, isolated_thp_path):
        self.isolated_thp = pd.read_csv(isolated_thp_path, header=0, index_col=0)
        self._logger.info('Isolated thp\n'
                        f'{self.isolated_thp}')


    def __parse_shared_thp(self, shared_thp_path):
        self.shared_thp = pd.read_csv(shared_thp_path, header=0, index_col=0)
        self._logger.info('Shared thp\n'
                        f'{self.shared_thp}')


    def __parse_contention_sensitivity(self, contention_sensitivity_path):
        self.contention_sensitivity = pd.read_csv(contention_sensitivity_path, header=0, index_col=0)
        self._logger.info('Contention sensitivity\n'
                        f'{self.contention_sensitivity}')


    def update_progress(self, job, samples):
        self._logger.info(f'Training job {job.id} samples:{samples}')

        # Update the number of trained samples so far
        job.trained_samples.update(samples)

        # Update system thp based on current round's thp
        isolated_thp = self.isolated_thp_get(job.workload)
        self.system_thp.update(samples/(isolated_thp*self.round_duration))

        # Round ended: queue the job if it isn't complete yet
        if job.trained_samples.sum < job.required_trained_samples:
            self.job_queue.put(job)


    def stopping_cond(self):
        no_resources = False
        queue_is_empty = self.job_queue.empty()
        return no_resources or queue_is_empty


    def schedule(self, cluster_nodes, job_queue):
        schedule = []

        # TODO: Iterate until cluster is full
        for i in range(int(len(cluster_nodes)/2)):
            if self.job_queue.empty():
                break

            # TODO: schedule jobs until node resources are exhausted
            job_a = self.job_queue.get()
            if len(self.job_queue) > 0:
                job_b = self.job_queue.get()
            else:
                job_b = None

            self._logger.info(f'Picked job[{job_a.id}] job[{job_b.id if job_b is not None else ""}]')

            # TODO: Find available nodes
            schedule.append({'joblist': [job_a, job_b], 'nodelist': [cluster_nodes[i*2], cluster_nodes[i*2+1]]})

        self.makespan.update(self.round_duration)

        return schedule


    def __str__(self):
        return '%s:%s:%s:%s' % (self.__class__, self.contention_aware, self.makespan.sum, self.system_thp.avg)