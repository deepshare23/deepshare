from simulator.node import Node
from simulator.job import parse_trace
import job_helper
import obs_helper


class DummyJob:
    def __init__(self, id, workload, gpu_demand, isolated_throughput, required_trained_samples):
        self.id = id # int
        self.workload = workload # model
        self.total_gpu_demand = gpu_demand
        self.num_nodes_to_schedule = 0
        self.per_node_gpu_demand = gpu_demand
        self.isolated_throughput = isolated_throughput
        self.required_trained_samples = required_trained_samples
        self.trained_samples = 0

    def __str__(self):
        return f'Dummy Job: {self.id} {self.workload}, total demand: {self.total_gpu_demand} per-node demand: {self.per_node_gpu_demand} placement: {self.num_nodes_to_schedule}'


# Test cases
if __name__ == '__main__':
    job0 = DummyJob(0, 'MobileNetV3', 2, -1, 0)
    job1 = DummyJob(1, 'FSDP', 4, -1, 0)
    job2 = DummyJob(2, 'MoE', 8, -1, 0)
    jobs = [job0, job1, job2]

    job_key0 = f'{job0.workload}-{job0.total_gpu_demand}'
    job_key1 = f'{job1.workload}-{job1.total_gpu_demand}'
    job_key2 = f'{job2.workload}-{job2.total_gpu_demand}'
    job_keys = [job_key0, job_key1, job_key2]
    job_sns = [job_helper.get_sn(job0), job_helper.get_sn(job1), job_helper.get_sn(job2)]

    node1 = Node(1, 8)
    node2 = Node(2, 8)
    job0.per_node_gpu_demand = int(job0.total_gpu_demand)
    job1.per_node_gpu_demand = int(job1.total_gpu_demand)
    job2.per_node_gpu_demand = int(job2.total_gpu_demand)
    node1.schedule_job(job0)
    node1.schedule_job(job1)
    node2.schedule_job(job2)
    nodes = [node1, node2]

    print(f"===== Test: compute_possible_placement of Job {job_sns[0]} ({job_key0}) Job {job_sns[1]} ({job_key1}) Job {job_sns[2]} ({job_key2})")
    for job in jobs:
        obs_helper.compute_possible_placement(job)

    print(f'===== Test: compute_cluster_state')
    obs_helper.compute_cluster_state(nodes)

    node1.remove_job(job1)
    print(f'===== Test: compute_cluster_state after removing Job {job_sns[1]} ({job_key1})')
    obs_helper.compute_cluster_state(nodes)
