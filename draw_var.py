import matplotlib.pyplot as plt
import numpy as np
bit = [4, 5, 6, 7, 8]
# 0 iteration
# bhq = [1.3727991580963135, 0.2963806986808777, 0.04555100202560425, 0.01297694444656372, 0.004160284996032715]
# psq = [2.457460641860962, 0.4441693425178528, 0.08043527603149414, 0.017490267753601074, 0.00601726770401001]
# ptq = [4.693792819976807, 0.82042479515075684, 0.14638060331344604, 0.03927326202392578, 0.014725446701049805]

# # 100 iteration
# bhq = [0.026774883270263672, 0.008239150047302246, 0.0019496679306030273, 0.0008063316345214844, 0.0006188154220581055]
# psq = [0.18330025672912598, 0.05142045021057129, 0.013677418231964111, 0.0034197568893432617, 0.0011163949966430664]
# ptq = [1, 1.0757898092269897, 0.2793232202529907, 0.11141997575759888, 0.0555720329284668]
#
# ptq_two_all = [1, 1.307e+00, 3.187e-01, 1.509e-01, 7.056e-02]
# ptq_two_half = [1, 2.285e+00, 3.458e-01, 1.509e-01, 7.648e-02]
# 200 iteration
# bhq = [0.0008355081081390381, 0.00022105127573013306, 4.0046870708465576e-05, 4.6584755182266235e-05, 2.8811395168304443e-05]
# psq = [0.0102195143699646, 0.003302179276943207, 0.0010490566492080688, 0.0003215521574020386, 3.758817911148071e-05]
# ptq = [1, 0.489107608795166, 0.05248359590768814, 0.009677514433860779, 0.006047338247299194]

# 100 iteration second method raw

plt.figure()
plt.plot(bit, np.log10(ptq_raw_vari), label='var')
# plt.plot(bit, ptq_raw_bias, label='bias')
# plt.plot(bit, np.log10(bhq), label='bhq')
# plt.plot(bit, np.log10(psq), label='psq')
# plt.plot(bit, np.log10(ptq), label='ptq')
# plt.plot(bit, np.log10(ptq_two_all), label='ptq_two_all')
# plt.plot(bit, np.log10(ptq_two_half), label='ptq_two_half')
# plt.legend()
plt.savefig('results/grad_var.png')

plt.figure()
plt.plot(bit, np.log10(np.abs(bias)), label='bias_false')
plt.plot(bit, np.log10(np.abs(bias_two_all)), label='bias_two_all')
plt.plot(bit, np.log10(np.abs(bias_two_half)), label='bias_two_half')
plt.legend()
plt.savefig('results/grad_bias.png')