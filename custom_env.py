
from os import path 

from gymnasium import spaces 
import gymnasium as gym 
import numpy as np 

class SoftRobotic(gym.Env): 

    """
    ## Description 

    x,y     :the cartesian coordinate of the end point of the soft robotic arm 
            :Down is +y and Right is +x 

    theta   :the angle of the soft end point of the soft robotic arm from the centre axis 
            :Right from centre axis is +theta, left from centre axis is -theta 

    tau     :the torque applied to the soft robotic arm 
            :Torque that makes the arm turn right is +tau, left is -tau

    force_left, force_right :the force applied in the roboit arm in the +y direction 
    
    ## Action space 

    !!! To be implemented !!! 
    The data structure used to store the action space Eg 

    | Num | Action | Min  | Max |
    |-----|--------|------|-----|
    | 0   | Torque | -2.0 | 2.0 |

    ## Stage one action space: 
    Inlude only torque control 

    ## Stage two action space: 
    Include the control for KP KI KD of the PID controller 



    ## Obervation space 

    x,y     :The position of the end point of the soft robotic arm. 
        x_real,y_real 
        x_control,y_control 

    theta   :The angle of the end point of the soft robotics arm. 
        theta_real 
        theta_control
    
    





    
    """ 

    def _get_obs(self): 

    def _get_info(self): 

    def reset(self): 

    def step(self): 

            