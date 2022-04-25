
**Failure of Non-DL approach**

The accuracy of prediction of current position can be improved if past accelerations are used as input along side the current accelerations.I used Linear regression to experiment but didnt see any good enough results.Heuristic models gave better results if error was corrected very frequently when double intergration for accelerations were done and huge errors were observed when distances were summed to get total distance.

**Distance = Double Integration(Acceleration)**

This becomes a perfect use case for Recurrent Neural Networks and hence, LSTMs are chosen.



