import tensorflow as tf
from plotfunc import construct_joint3D, draw_mot
import numpy as np
import scipy.io as scio
from tensorflow import keras

keras.config.enable_unsafe_deserialization()

#epoch = [0,50,100]
epoch = np.linspace(0,150,4, dtype= int)
dim_sentence = 512
latent_dim = 20
case_number = 18
test_case = [246, 890, 1368]

#Test Set
#test_action = np.load('../Data/test_action.npy') #(6384, 32, 24)
test_script = np.load('../Data/test_script.npy') #(6384, 512)
#Validation Set
#validate_action = np.load('../Data/val_action.npy') 
#validate_script = np.load('../Data/val_script.npy')

init_pose = scio.loadmat('../Data/mean_pose.mat')['mean_vector']
init_pose = np.transpose(init_pose,(1,0))
init_input = tf.tile(tf.expand_dims(init_pose, axis=0), [1, 1, 1])
print(init_input.shape)
'''
def test_GAN(epoch):

    for test, idx in enumerate(test_case):

        real_motion = validate_action[[idx]]
        real_motion_transpose =np.transpose(real_motion,(0,2,1))
        script = validate_script[[idx]]

        real_action = construct_joint3D(real_motion_transpose)

        draw_mot(real_action,'Real', f'Real Action {idx}',
            f'../Hyperparameter_Tuning_and_Ablation_Study/Model_and_Results/Case_{case_number}/Real_{test}.gif')

        for i in epoch:

            generator = tf.keras.models.load_model(f'../Hyperparameter_Tuning_and_Ablation_Study/Model_and_Results/Case_{case_number}/model_epoch_{i}.keras')

            print(f'Using model {i}')

            script_convert = tf.convert_to_tensor(script, dtype=tf.float32)

            noise = tf.random.normal([1,latent_dim])
            generated_motion = generator.predict([noise,script_convert,init_input])

            print(script_convert.shape)
            print(generated_motion.shape)
            generated_motion = np.transpose(generated_motion,(0,2,1))
            print(generated_motion.shape)

            fake_action = construct_joint3D(generated_motion)

            draw_mot(fake_action,i, f'Fake Action {idx}',
            f'../Hyperparameter_Tuning_and_Ablation_Study/Model_and_Results/Case_{case_number}/Fake_{test}_{i}.gif')

test_GAN(epoch)
'''

Model_Epoch = 150
total_script_number = test_script.shape[0]
print(total_script_number)
generator = tf.keras.models.load_model(f'../Hyperparameter_Tuning_and_Ablation_Study/Model_and_Results/Case_{case_number}/model_epoch_{Model_Epoch}.keras')
total_motion_list = []

for i in range(total_script_number):

    print(i)
    script = test_script[[i]]
    #script_convert = tf.convert_to_tensor(script, dtype=tf.float32)

    noise = tf.random.normal([1,latent_dim])
    generated_motion = generator.predict([noise,script,init_input])

    total_motion_list.append(generated_motion[0]) 

total_motion_list = np.array(total_motion_list)   
print(total_motion_list.shape)
np.save(f'../Hyperparameter_Tuning_and_Ablation_Study/Model/Case_{case_number}/LSTM_Test_Motion_Model_Epoch_{Model_Epoch}.npy' ,total_motion_list)


'''
motion = np.load(f'../Hyperparameter_Tuning_and_Ablation_Study/Model/Case_{case_number}/LSTM_Test_Motion.npy')
cospeech_motion = np.load('../Baseline_Model/CoSpeech/exec/CoSpeech_Motion_List_Epoch_350.npy')
real_motion = np.load('../Data/test_action.npy')

motion_transpose =np.transpose(motion,(0,2,1))
cospeech_motion_transpose =np.transpose(cospeech_motion,(0,2,1))
real_motion_transpose =np.transpose(real_motion,(0,2,1))

for i in range(test_case):
    
    idx = np.random.randint(0, motion_transpose.shape[0])

    motion_idx = construct_joint3D(motion_transpose[idx])
    cospeech_motion_idx = construct_joint3D(cospeech_motion_transpose[idx])
    real_motion_transpose_idx = construct_joint3D(real_motion_transpose[idx])

    draw_mot(motion_idx,i, f'Generated Motion {idx}',
             f'../Results/Compare/Generated_Motion_{i}.gif')
    
    draw_mot(cospeech_motion_idx,i, f'Cospeech Motion {idx}',
             f'../Results/Compare/Cospeech_Motion_{i}.gif')
    
    draw_mot(real_motion_transpose_idx,i, f'Real Motion {idx}',
             f'../Results/Compare/Real_Motion_{i}.gif')
'''