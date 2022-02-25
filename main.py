import utils
from custom_pdf import CustomPDF
import matplotlib.pyplot as plt

if __name__ == "__main__":

    skewness_ = [0, 0, 0, 0]
    kurtosis_ = [2, 5, 10, 20]

    _, axs = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))

    for (s, k) in zip(skewness_, kurtosis_):

        print("Skewness = {}, Kurtosis = {}".format(s, k))

        kappa = utils.calculate_kappa(s, k)
        print("Kappa is {}".format(kappa))
        pdf = utils.calculate_probability_density_function(
            s, k)

        # utils.print_summary(pdf=pdf)
        # utils.print_moments(pdf=pdf)
        cpdf = CustomPDF(pdf=pdf)
        cpdf()
        data = cpdf.sample(n=10000)
        utils.print_summary(data=data)
        utils.print_moments(data=data)
        print("=" * 50)

        axs[0].plot(data, label="Skewness = {}, Kurtosis = {}, Roughness = {}".format(s, k, utils.calculate_roughness(data)))
        axs[1].hist(data, bins=50, density = True, stacked = True, 
                    label="Skewness = {}, Kurtosis = {}, Roughness = {}".format(s, k, utils.calculate_roughness(data)))
		
    plt.legend()
    plt.show()
