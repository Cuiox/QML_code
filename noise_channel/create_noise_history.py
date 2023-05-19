# create a list of T1, T2

import numpy as np
# from data import dump_load

def create_history_fake_t12(num_qubits, num_history, avg_t1, scale_t1, avg_t2, scale_t2):
    
    history_t1 = np.random.normal(loc = avg_t1, scale = scale_t1, size=(num_history, num_qubits))
    history_t2 = np.random.normal(loc = avg_t2, scale = scale_t2, size=(num_history, num_qubits))
    
    return history_t1, history_t2

def create_history_fake_depolarizing(num_qubits, num_history, avg_1q, scale_1q, avg_2q, scale_2q):
    '''avg_1q: [0. 0.005]'''
    history_1q = np.random.normal(loc = avg_1q, scale = scale_1q, size=(num_history, num_qubits))
    history_2q = np.random.normal(loc = avg_2q, scale = scale_2q, size=(num_history, num_qubits, num_qubits))
    
    history_1q[history_1q<=0] = avg_1q
    history_2q[history_2q<=0] = avg_2q
    
    # noise_infos = []
    
    return history_1q, history_2q

def create_noiseinfo_fake(num_qubits, **args):
    noise_info = {}
    if 'depolarizing' in args:
        if 'avg_1q' in args['depolarizing']:
            avg_1q = args['depolarizing'].get('avg_1q', 0.01)
            scale_1q = args['depolarizing'].get('scale_1q', avg_1q/2)
            _errors = np.random.normal(loc = avg_1q, scale = scale_1q, size=(num_qubits))
            _errors[_errors<=0] = avg_1q
            _errors[_errors>1.0] = 1.0
            noise_info['depolarizing'] = {
                'error_1q': _errors
            }
        elif 'range' in args['depolarizing']:
            start, end =  args['depolarizing']['range']
            assert end > start
            _errors = np.random.random(size=(num_qubits)) * (end-start) + start
            noise_info['depolarizing'] = {
                'error_1q': _errors
            }
                    
    return noise_info

def create_history_fake(num_qubits, num_history, **args):
    histories = []
    for _ in range(num_history):
        noise_info = create_noiseinfo_fake(num_qubits, **args)
        histories.append(noise_info)
    return histories
    
# path = f'history {num_qubits} {num_history} {avg_t1} {scale_t1} {avg_t2} {scale_t2}.pkl'

# def load_history(path):
#     return

if __name__ == '__main__':
    history_t1, history_t2 = create_history_fake_t12(num_qubits = 2, num_history= 10, avg_t1 = 118, scale_t1 = 3, avg_t2 = 32, scale_t2 = 3)
    print(history_t1, history_t2)
    
    history_1q, history_2q = create_history_fake_depolarizing(num_qubits = 2, num_history=10, avg_1q=0.002, scale_1q=0.002/4, avg_2q=0.01, scale_2q=0.01/5)
    print(history_1q, history_2q)
    