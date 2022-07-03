This repository contains decentralised AI training platform using Split Learning Private Computation Architecture using Django.

Methodology:
-The base deep learning CNN model is first created on the central server. This will serve as the baseline model of all of the further improvements. Then several instances of user clients will be created on different python notebooks to simulate different users in a network. 					
-After the training, the improved model is then saved onto the client directory. 					
-The new learnings of the CNN model is now sent back to the server, where the server incorporates the new model into its base model, if it perceives any significant improvement in the model. The server uses secure aggregation to do so. 	
-The new updated base model is now used as the base model for further communications and training on other devices of the network.


