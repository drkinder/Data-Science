class NaiveBayesTextClassifier:

    """ Naive Bayes Classifier geared towards text classification. Initialize Classifier with X, a 2d list comprised of
        lists of strings broken down into lists of individual words and y, a 1d list with the correct categorization of
        the texts at the same index as X.

        Call NaiveBayesTextClassifier.train() to train the model on the provided data.
        Call NaiveBayesTextClassifier.predict(test_x) to have the model return a a list of tuples containing the
        predicted category alongside the confidence of the model's prediction range 0-1.
        """

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.data_dict = {}

        self.universal_word_count = 0
        self.category_count = 0

    def predict(self, test_x):
        """ Predicts classification of provided test_x text. test_x should be a 2d list comprised of a list of strings
            broken down into lists of individual words. Method will return a list of tuples
            [('Predicted Category', prediction confidence 0-1)]. """

        """ P(A|w1∩w2∩w3∩...wn) = P(w1|A)*P(w2|A)*P(w3|A)*P(wn|A)*P(A) / 
                                   P(w1|A)*P(w2|A)*P(w3|A)*P(wn|A)*P(A) + 
                                   P(w1|B)*P(w2|B)*P(w3|B)*P(wn|B)*P(B)            """

        y_pred = []

        for x in test_x:
            posteriors = {}
            for category in self.data_dict:
                prior = self.get_prior(category)
                likelihood = self.get_likelihood(category, x)

                numerator = likelihood*prior
                denominator = numerator

                for cat in self.data_dict:
                    if cat != category:
                        denominator += self.get_likelihood(cat, x)*self.get_prior(cat)

                posteriors[category] = numerator / denominator

            pred = max(posteriors, key=posteriors.get)
            y_pred.append((pred, posteriors[pred]))

        return y_pred

    def get_prior(self, category):
        """ Returns the prior. P(category = True) """
        return self.data_dict[category]['data']['category_occurrence']/len(self.X)

    def get_likelihood(self, category, test_word_list):
        """ Returns the likelihood of each word in test_text given category is true:
            P(word1|category = True)*P(word2|category = True)*P(wordn|category = True) """

        likelihood = 0

        for word in test_word_list:
            # Zero frequency problem
            if word.lower() not in self.data_dict[category]['words']:
                freq = 1 / (self.universal_word_count + self.category_count)  # Add one to all values
            else:
                freq = ((self.data_dict[category]['words'][word.lower()] + 1) /
                        (self.universal_word_count + self.category_count))

            # Initialize likelihood - Cannot start multiplying to zero
            if likelihood == 0:
                likelihood = freq
            else:
                likelihood *= freq

        return likelihood

    def train(self):
        """ Trains the model on the provided data. """

        self.get_data_dict()
        self.set_constants()

    def set_constants(self):
        """ Correctly sets constants after self.get_data_dict is called. """

        for category in self.data_dict:
            self.universal_word_count += self.data_dict[category]['data']['total_word_count']

        self.category_count = len(self.data_dict)

    def get_data_dict(self):
        """ Populates self.data_dict with new dictionaries for each category in X under self.data_dict[category] to
            store the frequency of each word's use in the entire text pool for each category. A word's frequency can be
            accessed via self.data_dict[category][word]. """

        for idx, category in enumerate(self.y):

            if category not in self.data_dict:
                self.data_dict[category] = {'data': {'total_word_count': 0, 'category_occurrence': 1}, 'words': {}}
            else:
                self.data_dict[category]['data']['category_occurrence'] += 1  # count total number of cat. occurrences

            # Update word counts given new text data
            self.update_category_word_count(category, self.X[idx])

    def update_category_word_count(self, category, word_list):
        """ Populates provided category's associated dictionary with a counter for each word. Also updates the
            'total_word_count' """

        for word in word_list:

            if word.lower() not in self.data_dict[category]:
                self.data_dict[category]['words'][word.lower()] = 1
            else:
                self.data_dict[category]['words'][word.lower()] += 1

            self.data_dict[category]['data']['total_word_count'] += 1
