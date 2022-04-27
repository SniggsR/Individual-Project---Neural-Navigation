
**Failure of Non-DL approach**

The accuracy of prediction of current position can be improved if past accelerations are used as input along side the current accelerations.Past Data is necessary to understand from which positions we have to come to current state and that data is necessary to accurately keep our predictions on track without many deviations from noises.Regardless, I experimented with Linear regression but didnt see  good results.Heuristic models gave better results if error was corrected very frequently when double intergration for accelerations were done and huge errors were observed when distances were summed to get total distance.

**Distance = Double Integration(Acceleration)**

This becomes a perfect use case for Recurrent Neural Networks and hence, LSTMs are chosen.



