I have decided to implement the features of the alphgo zero paper for a game like tic tac toe.....



First lets discuss about the neural newtork:
-> In the alphago paper they had 19*19*17 here we will have 3*3*5 where each player will have 2 hisotry planes summing up to 4 planes and a current player plane
-> Then deciding the batch size as this game is very small so basically even batch_size=512 is also possible but lets go with the traditional way of having batch size=64
->Now lets go into the first convolutional layer whihc has 256 filters in the alphago paper . We absolutely don’t need to keep 256 filters here if we try that, our network will be hilariously overpowered for such a tiny game.It will still learn, but it’s like using a rocket engine to toast bread. 
->The kerenel_size=3*3 and stride=1
->Batch Normalizationa and then RELU activation