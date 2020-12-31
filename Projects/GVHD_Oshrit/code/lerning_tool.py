import pandas as pd
import numpy as np
from dateutil import relativedelta
from scipy.optimize import root, root_scalar

from Projects.GVHD_Oshrit.code.preprocces import add_tag_to_predict
from Projects.GVHD_Oshrit.code.utils import in_debug


def last_samples_before_gvhd(otu: pd.DataFrame, censored: bool, all=False) -> pd.DataFrame:
    """
    The func returns a new data frame which consists of only the last sample of the censored samples
    or- the last sample taken before GVHD event for the uncensored samples
    :param otu: otu merged data frame
    :return: data frame with only the last relevant samples for each person
    """
    last_samples_before_gvhd = pd.DataFrame()
    for subject in otu.groupby(otu["subjid"]):
        if censored is False:
            # Uncensored
            subject_no_sampels_after_cgvhd = subject[1].loc[
                pd.to_numeric(pd.to_datetime(subject[1]["DATE"]) - subject[1]["date_ofttcgvhd"]) < 0]
            if all:
                last_test = subject_no_sampels_after_cgvhd.T[subject_no_sampels_after_cgvhd.index.sort_values()].T
            else:
                last_test = subject_no_sampels_after_cgvhd.T[subject_no_sampels_after_cgvhd.index.sort_values()].T.tail(
                    1)
        else:
            # Censored
            # Take the last sample that exists
            if all:
                last_test = subject[1].T[subject[1].index.sort_values()].T
            else:
                last_test = subject[1].T[subject[1].index.sort_values()].T.tail(1)

        last_samples_before_gvhd = last_samples_before_gvhd.append(last_test)

    return last_samples_before_gvhd


class Augment(object):
    """
    This class is for the data augmentation for the censored samples.
    """

    def __init__(self, otu: pd.DataFrame, tag: str, censor_df: pd.DataFrame, uncensord_df: pd.DataFrame, beta,
                 bacteria_col_keyword="k__Bacteria", only_last_sample_before_gvhd=True):
        """

        :param otu: otu df with all features and tag
        :param tag: name of tag
        :param censor_df: data frame of only censored samples
        :param uncensord_df: data frame of only uncensored samples
        :param bacteria_col_keyword: a sign for all the columns of bacterias in order to drop the other
                                     irrelevant columns of the sample.
        """
        self.beta = beta
        self.tag = tag
        self.censored_org_tag = censor_df[tag]
        self.uncensored_org_tag = uncensord_df[tag]

        # remove all columns which are not bacterias:
        self.last_samples_censord = last_samples_before_gvhd(censor_df, censored=True)
        self.last_samples_censord_only_microbiom = self.last_samples_censord[
            [col for col in otu.columns if bacteria_col_keyword in col]]
        self.last_censord_tag_before_gvhd = censor_df[tag].loc[self.last_samples_censord.index]

        self.last_samples_uncensord = last_samples_before_gvhd(uncensord_df, censored=False,
                                                               all=not only_last_sample_before_gvhd)
        self.last_samples_uncensord_only_microbiom = self.last_samples_uncensord[
            [col for col in otu.columns if bacteria_col_keyword in col]]
        self.last_uncensord_tag_before_gvhd = uncensord_df[tag].loc[self.last_samples_uncensord.index]

    def augment(self, beta=0.01):
        raise NotImplementedError

    def add_tag_to_predict(self, censored_df, uncensored_df):
        raise NotImplementedError

    def augment_fix(self, artificial_time):
        """
        chooses the latest time for event time for censored samples
        :param artificial_time: the result of augment, sereies with artificial time
        :return: data frame with logical event time to censored samples
        """
        fixed_augment = \
            self.last_samples_censord.loc[
                self.last_samples_censord[self.tag].sort_index() >= artificial_time.sort_index()][
                self.tag]
        good_artificials = artificial_time.loc[
            self.last_samples_censord[self.tag].sort_index() < artificial_time.sort_index()]

        # prints the number of logical artificial samples vs unlogical ones
        if in_debug():
            print("Bad artificials: " + str(len(fixed_augment)))
            print("Good artificials: " + str(len(good_artificials)))
            print("Percent of good artificials: %" + str((len(good_artificials) / len(artificial_time)) * 100))
        self.good_frac = len(good_artificials) / len(artificial_time)
        return fixed_augment.append(good_artificials)

    def implement_augment(self, censored_df):
        """
        creates the new column of new time for the censored samples. The func adds the time to all samples
        by taking the time we created to the last sample.
        :type org_df: censored saliva otu or censored stool otu with all samples
        :param censored_df: .censored saliva otu or censored stool otu with all samples
        :param artificial_data: new time for censored samples.
        :param tag: name of tag
        :return: new censored df with the new column
        """
        artificial_data_for_censored_sample = self.augment(self.beta)

        censored_df["new_" + self.tag] = artificial_data_for_censored_sample
        for subject in censored_df.groupby(censored_df["subjid"]):
            aug_index = set(subject[1].index).intersection(artificial_data_for_censored_sample.index)
            subject[1]["new_" + self.tag] = artificial_data_for_censored_sample[aug_index].values[0]
            censored_df.update(subject[1])
        return censored_df


class Bar_Augment(Augment):
    """
    the last_uncensored_tag_before_gvhd is the time in years from the bmt
    (transplant date to appearance of GVHD.
    """

    def augment(self, beta=0.01) -> pd.Series:
        """
        calculate the GVHD event time for the censored samples.
        calculation follows: time_i = (sum for all uncensored samples on d_i,j * time_j) / (sum for all uncensored samples on d_i,j)
        while d_i,j is: e ^ ( -beta*(norm_l2(sample_i - sample_j))^2)
        while i is an index of censored sample and j is an index of the uncensored sample
        :param beta:
        :return: series with artificial times
        """
        artificial_time = pd.Series()
        for x_line in self.last_samples_censord_only_microbiom.iloc:
            Y = self.last_samples_uncensord_only_microbiom.sub(x_line, axis=1)
            Y = np.square(Y).sum(axis=1)
            Y = Y * (-beta)
            di = np.exp(Y)

            line = pd.Series({x_line.name: (di * self.last_uncensord_tag_before_gvhd).sum() / di.sum()})
            artificial_time = artificial_time.append(line)

        return self.augment_fix(artificial_time)

    def add_tag_to_predict(self, censored_df, uncensored_df):
        """
        add a tag of the time passes from the sample time to the event in uncensored:
        and in censored: the time passes from the sample time to the augmented event and a column for the time passes
        from the sample to competing event.
        :param censor_df: data frame of censor
        :param uncensor_df: data frame of uncensored
        :param tag: name of tag
        :return: censor_df, uncensor_df
        """
        # add column tag to censored
        censored_df["time_to_" + self.tag] = ((pd.to_datetime(censored_df["bmtdate"]) + (
                censored_df["new_" + self.tag] * 365.25).apply(
            lambda x: relativedelta.relativedelta(days=x))) - pd.to_datetime(censored_df["DATE"])) / pd.to_timedelta(
            1,
            unit='D')

        # add column  for loss to censored
        censored_df[self.tag + "_for_loss"] = ((pd.to_datetime(censored_df["bmtdate"]) + (
                censored_df[self.tag] * 365.25).apply(
            lambda x: relativedelta.relativedelta(days=x))) - pd.to_datetime(censored_df["DATE"])) / pd.to_timedelta(1,
                                                                                                                     unit='D')
        # add column to tag to uncensored
        uncensored_df["time_to_" + self.tag] = ((pd.to_datetime(uncensored_df["bmtdate"]) + (
                uncensored_df[self.tag] * 365.25).apply(
            lambda x: relativedelta.relativedelta(days=x))) - pd.to_datetime(uncensored_df["DATE"])) / pd.to_timedelta(
            1,
            unit='D')
        return censored_df, uncensored_df


class Lozon_Augment(Augment):
    def __init__(self, otu: pd.DataFrame, tag: str, censor_df: pd.DataFrame, uncensord_df: pd.DataFrame, beta):
        self.tag = tag
        censored_df, uncensored_df = self.preprocces_data(censor_df, uncensord_df)
        self.censored_tag_for_loss = censored_df[tag + "_for_loss"]
        self.uncensored_time_to_tag = uncensored_df["time_to_" + tag]
        super().__init__(otu, f"time_to_{tag}", censored_df, uncensored_df, beta)

    def preprocces_data(self, censor_df, uncensor_df):
        """
        add a tag of the time passes from the sample time to the event in uncensored:
        and in censored: the time passes from the sample time to the augmented event and a column for the time passes
        from the sample to competing event.
        :param censor_df: data frame of censor
        :param uncensor_df: data frame of uncensored
        :param tag: name of tag
        :return: censor_df, uncensor_df
        """
        # add column tag to censored
        censor_df["time_to_" + self.tag] = ((pd.to_datetime(censor_df["bmtdate"]) + (
                censor_df[self.tag] * 365.25).apply(lambda x: relativedelta.relativedelta(days=x))) - pd.to_datetime(
            censor_df["DATE"])) / pd.to_timedelta(1, unit='D')
        # add column  for loss to censored
        censor_df[self.tag + "_for_loss"] = ((pd.to_datetime(censor_df["bmtdate"]) + (
                censor_df[self.tag] * 365.25).apply(lambda x: relativedelta.relativedelta(days=x))) - pd.to_datetime(
            censor_df["DATE"])) / pd.to_timedelta(1, unit='D')
        # add column to tag to uncensored
        uncensor_df["time_to_" + self.tag] = ((pd.to_datetime(uncensor_df["bmtdate"]) + (
                uncensor_df[self.tag] * 365.25).apply(lambda x: relativedelta.relativedelta(days=x))) - pd.to_datetime(
            uncensor_df["DATE"])) / pd.to_timedelta(1, unit='D')
        return censor_df, uncensor_df

    def augment(self, beta=0.01) -> pd.Series:
        """
        calculate the GVHD event time for the censored samples.
        calculation follows: time_i = (sum for all uncensored samples on d_i,j * time_j) / (sum for all uncensored samples on d_i,j)
        while d_i,j is: e ^ ( -beta*(norm_l2(sample_i - sample_j))^2)
        while i is an index of censored sample and j is an index of the uncensored sample
        :param beta:
        :return: series with artificial times
        """
        artificial_time = pd.Series()

        for x_line in self.last_samples_censord_only_microbiom.iloc:
            Y = self.last_samples_uncensord_only_microbiom.sub(x_line, axis=1)
            Y = np.square(Y).sum(axis=1)
            Y = Y * (-beta)
            di = np.exp(Y)

            line = pd.Series({x_line.name: (di * self.last_uncensord_tag_before_gvhd).sum() / di.sum()})
            artificial_time = artificial_time.append(line)

        return self.augment_fix(artificial_time)

    def implement_augment(self, censored_df):
        """
        creates the new column of new time for the censored samples. The func adds the time to all samples
        by taking the time we created to the last sample.
        :type org_df: censored saliva otu or censored stool otu with all samples
        :param censored_df: .censored saliva otu or censored stool otu with all samples
        :param artificial_data: new time for censored samples.
        :param tag: name of tag
        :return: new censored df with the new column
        """
        artificial_data_for_censored_sample = self.augment(self.beta)

        censored_df["new_" + self.tag] = artificial_data_for_censored_sample
        censored_df["DATE_LAST_SAMPLE"] = self.last_samples_censord["DATE"]
        for subject in censored_df.groupby(censored_df["subjid"]):
            aug_index = set(subject[1].index).intersection(artificial_data_for_censored_sample.index)
            subject[1]["new_" + self.tag] = artificial_data_for_censored_sample[aug_index].values[0]
            subject[1]["DATE_LAST_SAMPLE"] = subject[1].loc[sorted(subject[1].index)[-1]]["DATE"]

            censored_df.update(subject[1])
        return censored_df

    def add_tag_to_predict(self, censored_df, uncensored_df):
        censored_df[self.tag.replace("time_to_", "") + "_for_loss"] = self.censored_tag_for_loss
        uncensored_df[self.tag] = self.uncensored_time_to_tag
        censored_df[self.tag] = (pd.to_datetime(censored_df["DATE_LAST_SAMPLE"]) - pd.to_datetime(
            censored_df["DATE"])).apply(lambda x: x.days) + censored_df["new_" + self.tag]

        return censored_df, uncensored_df


class Bar_thesis_Augment(Augment):
    def __init__(self, otu: pd.DataFrame, tag: str, censor_df: pd.DataFrame, uncensord_df: pd.DataFrame, beta):
        self.tag = tag
        censored_df, uncensored_df = self.preprocces_data(censor_df, uncensord_df)
        self.censored_tag_for_loss = censored_df[tag + "_for_loss"]
        self.uncensored_time_to_tag = uncensored_df["time_to_" + tag]
        super().__init__(otu, f"time_to_{tag}", censored_df, uncensored_df, beta, only_last_sample_before_gvhd=False)

    def preprocces_data(self, censor_df, uncensor_df):
        """
        add a tag of the time passes from the sample time to the event in uncensored:
        and in censored: the time passes from the sample time to the augmented event and a column for the time passes
        from the sample to competing event.
        :param censor_df: data frame of censor
        :param uncensor_df: data frame of uncensored
        :param tag: name of tag
        :return: censor_df, uncensor_df
        """
        # add column tag to censored
        censor_df["time_to_" + self.tag] = ((pd.to_datetime(censor_df["bmtdate"]) + (
                censor_df[self.tag] * 365.25).apply(lambda x: relativedelta.relativedelta(days=x))) - pd.to_datetime(
            censor_df["DATE"])) / pd.to_timedelta(1, unit='D')
        # add column  for loss to censored
        censor_df[self.tag + "_for_loss"] = ((pd.to_datetime(censor_df["bmtdate"]) + (
                censor_df[self.tag] * 365.25).apply(lambda x: relativedelta.relativedelta(days=x))) - pd.to_datetime(
            censor_df["DATE"])) / pd.to_timedelta(1, unit='D')
        # add column to tag to uncensored
        uncensor_df["time_to_" + self.tag] = ((pd.to_datetime(uncensor_df["bmtdate"]) + (
                uncensor_df[self.tag] * 365.25).apply(lambda x: relativedelta.relativedelta(days=x))) - pd.to_datetime(
            uncensor_df["DATE"])) / pd.to_timedelta(1, unit='D')
        return censor_df, uncensor_df

    def augment(self, beta=0.01) -> pd.Series:
        """
        calculate the GVHD event time for the censored samples.
        calculation follows: time_i = (sum for all uncensored samples on d_i,j * time_j) / (sum for all uncensored samples on d_i,j)
        while d_i,j is: e ^ ( -beta*(norm_l2(sample_i - sample_j))^2)
        while i is an index of censored sample and j is an index of the uncensored sample
        :param beta:
        :return: series with artificial times
        """
        artificial_time = pd.Series()

        for x_line in self.last_samples_censord_only_microbiom.iloc:
            Y = self.last_samples_uncensord_only_microbiom.sub(x_line, axis=1)
            Y = np.sqrt(np.square(Y).sum(axis=1))
            Y = Y * (-beta)
            di = np.exp(Y)

            line = pd.Series({x_line.name: (di * self.last_uncensord_tag_before_gvhd).sum() / di.sum()})
            artificial_time = artificial_time.append(line)

        return self.augment_fix(artificial_time)

    def add_tag_to_predict(self, censored_df, uncensored_df):
        censored_df[self.tag.replace("time_to_", "") + "_for_loss"] = self.censored_tag_for_loss
        uncensored_df[self.tag] = self.uncensored_time_to_tag
        censored_df[self.tag] = (pd.to_datetime(censored_df["DATE_LAST_SAMPLE"]) - pd.to_datetime(
            censored_df["DATE"])).apply(lambda x: x.days) + censored_df["new_" + self.tag]

        return censored_df, uncensored_df

    def implement_augment(self, censored_df):
        """
        creates the new column of new time for the censored samples. The func adds the time to all samples
        by taking the time we created to the last sample.
        :type org_df: censored saliva otu or censored stool otu with all samples
        :param censored_df: .censored saliva otu or censored stool otu with all samples
        :param artificial_data: new time for censored samples.
        :param tag: name of tag
        :return: new censored df with the new column
        """
        artificial_data_for_censored_sample = self.augment(self.beta)

        censored_df["new_" + self.tag] = artificial_data_for_censored_sample
        censored_df["DATE_LAST_SAMPLE"] = self.last_samples_censord["DATE"]
        for subject in censored_df.groupby(censored_df["subjid"]):
            aug_index = set(subject[1].index).intersection(artificial_data_for_censored_sample.index)
            subject[1]["new_" + self.tag] = artificial_data_for_censored_sample[aug_index].values[0]
            subject[1]["DATE_LAST_SAMPLE"] = subject[1].loc[sorted(subject[1].index)[-1]]["DATE"]

            censored_df.update(subject[1])
        return censored_df


class MLE_Augment(Bar_thesis_Augment):
    def __init__(self, otu: pd.DataFrame, tag: str, censor_df: pd.DataFrame, uncensord_df: pd.DataFrame, beta):
        super().__init__(otu, tag, censor_df, uncensord_df, beta)
        self.TC = super().augment(0.1)

    def MLE_func_prop_dev(self, t):
        return t - self.tc + np.e ** (-self.lamda * t) * (self.sigma ** 2 - t + self.tc)

    def MLE_func(self, t):
        return -(((t - self.tc) / self.sigma) ** 2) + (
                (self.lamda * np.e ** (-self.lamda * t)) / (1 - (np.e ** (-self.lamda * t))))

    def augment(self, lamda):

        self.lamda = 0.001

        t = pd.Series()
        for i in self.TC.index:
            self.tc = self.TC[i]
            t = t.append(pd.Series({i: root_scalar(self.MLE_func, x0=self.tc, x1=1).root}))
        return self.augment_fix(t.astype(dtype="float"))

    def implement_augment(self, censored_df, lamda=0.01):
        """
        creates the new column of new time for the censored samples. The func adds the time to all samples
        by taking the time we created to the last sample.
        :type org_df: censored saliva otu or censored stool otu with all samples
        :param censored_df: .censored saliva otu or censored stool otu with all samples
        :param artificial_data: new time for censored samples.
        :param tag: name of tag
        :return: new censored df with the new column
        """
        self.bar_good_frac = self.good_frac
        self.sigma = self.TC.values.mean()*2
        artificial_data_for_censored_sample = self.augment(lamda)

        censored_df["new_" + self.tag] = artificial_data_for_censored_sample
        censored_df["DATE_LAST_SAMPLE"] = self.last_samples_censord["DATE"]
        for subject in censored_df.groupby(censored_df["subjid"]):
            aug_index = set(subject[1].index).intersection(artificial_data_for_censored_sample.index)
            subject[1]["new_" + self.tag] = artificial_data_for_censored_sample[aug_index].values[0]
            subject[1]["DATE_LAST_SAMPLE"] = subject[1].loc[sorted(subject[1].index)[-1]]["DATE"]

            censored_df.update(subject[1])
        return censored_df
