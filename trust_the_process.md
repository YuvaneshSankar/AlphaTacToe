I have decided to implement the features of the alphgo zero paper for a game like tic tac toe.....



First lets discuss about the neural newtork:
-> In the alphago paper they had 19*19*17 here we will have 3*3*5 where each player will have 2 hisotry planes summing up to 4 planes and a current player plane . Then deciding the batch size as this game is very small so basically even batch_size=512 is also possible but lets go with the traditional way of having batch size=64
->Now lets go into the first convolutional layer whihc has 256 filters in the alphago paper . We absolutely don’t need to keep 256 filters here if we try that, our network will be hilariously overpowered for such a tiny game.It will still learn, but it’s like using a rocket engine to toast bread.SO lets keep 32 filters. The kerenel_size=3*3 and stride=1 . Batch Normalizationa and then RELU activation
->Then comes our residual tower with convolutional layer of 32 filters with 3*3 kernel and stride=1 .  Then we have batch normalization RELU again convlutional layer Batch normalization then we have skip connection finishing with RELU. They have used 19 04 39 residual blocks lets use 10.
->Then we are splitting the network into policy head and value head 
->Policy head is where we have a convolutional layer with 2 filters and stride=1 where kernel is 1*1 . Then batch normalization then relu then just flattent the features from 3*3*2 to 18 . Then we have a fully connected layer with outptu layer showing 9 output probs represeting each point int the board. Then apply softmax.
->Now lets talk about value head same 1*1 conv with 1 filter and stride=1. Batch normalization then relu then flatten it into 9 as we get the ouptut as 3*3*1 from the conv layer.Then a fully connected layer expecting (with say 50 hidden units) and a tanh activation function.


!!!!!!!!!!I have completed the neural network arch

so whats next lets start with the monteCarloTreeSearch
