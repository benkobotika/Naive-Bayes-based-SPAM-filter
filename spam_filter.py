import re
import math


# all file names (returns ham, spam)
def readData(fileName):
    ham = []
    spam = []
    with(open(fileName, 'r')) as file:
        for line in file:
            line = line.rstrip('\n')
            if '.ham.' in line:
                ham.append(line)
            else:
                spam.append(line)
    return ham, spam


# read all stop words
def readStopData(fileName):
    stopWords = []
    with(open(fileName, 'r')) as file:
        for line in file:
            line = line.rstrip('\n')
            stopWords.append(' ' + line + ' ')
    return stopWords


# preprocess a mail (lowercase, stop-word filter, split to words)
def preprocess(mail, stopWords):
    with(open('all/' + mail, 'r', encoding='ISO-8859-1')) as file:
        newMail = file.read()
        newMail = newMail[9:]  # delete "subject:"

        # lowerCase
        newMail = newMail.lower()

        # delete uninterested words
        newMail = re.sub(r'[,.:!@#$%^&*()]', '', newMail)
        newMail = re.sub(r'\s+', ' ', newMail)

        words = newMail.split(' ')

        words = [w for w in words if w not in stopWords]

    appearance = {}
    for word in words:
        count = words.count(word)
        if word not in appearance:
            appearance[word] = count

    return appearance, words, len(words)


# preprocess all emails
def allPreprocess(data, stopWords):
    allAppearance = {}
    allWordsCount = 0
    for i in data:
        appearance, _, wordsCount = preprocess(i, stopWords)
        allWordsCount += wordsCount
        for word in appearance:
            count = appearance[word]
            if word not in allAppearance:
                allAppearance[word] = count
            else:
                allAppearance[word] += count

    return allAppearance, allWordsCount


# predicting one email (spam/ham)
def predictData(email, stopWords, pSpam, pHam, spamDictionary, hamDictionary,
                allWordsCountSpam, allWordsCountHam, alfa, V):
    _, content, _ = preprocess(email, stopWords)
    l = 0.00000001

    lnR = (math.log(pSpam) - math.log(pHam))
    for word in content:
        if word in spamDictionary:
            pWSpam = (spamDictionary[word] + alfa) / (alfa * V + allWordsCountSpam)
        else:
            pWSpam = alfa / (alfa * V + allWordsCountSpam)
            if pWSpam == 0:
                pWSpam = l

        if word in hamDictionary:
            pWHam = (hamDictionary[word] + alfa) / (alfa * V + allWordsCountHam)
        else:
            pWHam = alfa / (alfa * V + allWordsCountHam)
            if pWHam == 0:
                pWHam = l

        lnR += math.log(pWSpam) - math.log(pWHam)

    # R = math.e ** lnR
    # spamProbability = R / (R + 1)
    # hamProbability = 1 / (R + 1)

    return lnR > 0


# predicting all emails and add predict labels to all
def predictAllData(emailNames, stopWords, pSpam, pHam, spamDictionary, hamDictionary,
                   allWordsCountSpam, allWordsCountHam, alfa, V):
    spamCount = 0
    predictLabel = []
    i = 0
    for email in emailNames:
        if predictData(email, stopWords, pSpam, pHam, spamDictionary, hamDictionary,
                       allWordsCountSpam, allWordsCountHam, alfa, V):
            predictLabel.append((email, True))
        else:
            predictLabel.append((email, False))
        i += 1

    return predictLabel


# calculate the accuracy based on predicted labels and real labels (and spam falseNegative and falsePositive accuracy)
def accuracy(predictLabel):
    ok = 0
    # for spam
    falsePositive = 0
    falseNegative = 0
    spamCount = 0
    hamCount = 0
    for i in predictLabel:
        (emailName, prediction) = i
        if '.spam.' in emailName:
            if prediction:
                ok += 1
            else:  # predicts as ham but it's spam
                falseNegative += 1
            spamCount += 1
        else:
            if not prediction:
                ok += 1
            else:  # predicts as spam but it's ham
                falsePositive += 1
            hamCount += 1

    return ok / len(predictLabel), falseNegative / spamCount, falsePositive / hamCount


# classification, error rate, false pos-neg
def classificationErrorRateFalsePosNeg(trainHamNames, testHamNames, trainSpamNames, testSpamNames,
                                       trainSpamDictionary, trainHamDictionary,
                                       allWordsCountTrainHam, allWordsCountTrainSpam,
                                       allWordsCountTestHam, allWordsCountTestSpam,
                                       stopWords, pSpam, pHam, alfa, V):
    predictLabelTrainHamNames = predictAllData(trainHamNames, stopWords, pSpam, pHam,
                                               trainSpamDictionary, trainHamDictionary,
                                               allWordsCountTrainSpam, allWordsCountTrainHam, alfa, V)
    predictLabelTestHamNames = predictAllData(testHamNames, stopWords, pSpam, pHam,
                                              trainSpamDictionary, trainHamDictionary,
                                              allWordsCountTestSpam, allWordsCountTestHam, alfa, V)
    predictLabelTrainSpamNames = predictAllData(trainSpamNames, stopWords, pSpam, pHam,
                                                trainSpamDictionary, trainHamDictionary,
                                                allWordsCountTrainSpam, allWordsCountTrainHam, alfa, V)
    predictLabelTestSpamNames = predictAllData(testSpamNames, stopWords, pSpam, pHam,
                                               trainSpamDictionary, trainHamDictionary,
                                               allWordsCountTestSpam, allWordsCountTestSpam, alfa, V)

    # learning and testing error rate
    # accuracy for train data
    predictLabelTrain = predictLabelTrainHamNames + predictLabelTrainSpamNames
    trainAccuracy, _, _ = accuracy(predictLabelTrain)
    print("Accuracy for train data:")
    print(trainAccuracy)
    print("Error for train data:")
    print(1 - trainAccuracy)

    # accuracy for test data
    predictLabelTest = predictLabelTestHamNames + predictLabelTestSpamNames
    testAccuracy, _, _ = accuracy(predictLabelTest)
    print()
    print("Accuracy for test data:")
    print(testAccuracy)
    print("Error for test data:")
    print(1 - testAccuracy)

    # false positive and false negative for spam class
    _, falseNegative, falsePositive = accuracy(predictLabelTrain)
    print()
    print("For spam class (train):")
    print("false-negative=", falseNegative)
    print("false-positive=", falsePositive)

    _, falseNegative, falsePositive = accuracy(predictLabelTest)
    print()
    print("For spam class (test):")
    print("false-negative=", falseNegative)
    print("false-positive=", falsePositive)


def main():
    print('Reading data...')
    trainHamNames, trainSpamNames = readData('train.txt')
    testHamNames, testSpamNames = readData('test.txt')
    stopWords = readStopData('stopwords.txt')
    stopWords2 = readStopData('stopwords2.txt')
    stopWords = stopWords + stopWords2

    print('Training model...')
    # parameter estimation based on training data
    pSpam = len(trainSpamNames) / (len(trainSpamNames) + len(trainHamNames))
    pHam = 1 - pSpam
    trainHamDictionary, allWordsCountTrainHam = allPreprocess(trainHamNames, stopWords)  # word: appearance, all wordNr
    trainSpamDictionary, allWordsCountTrainSpam = allPreprocess(trainSpamNames, stopWords)
    testHamDictionary, allWordsCountTestHam = allPreprocess(testHamNames, stopWords)
    testSpamDictionary, allWordsCountTestSpam = allPreprocess(testSpamNames, stopWords)

    V = len({**trainHamDictionary, **trainSpamDictionary})  # for additive smoothing

    # classification
    # emails to test
    check = input('Do you want to check accuracy? ("1" if YES, "0" if NO) : ')
    if check == '1':
        print("Without additive smoothing factor: ")
        alfa = 0  # doesn't matter (managed) => p = l
        classificationErrorRateFalsePosNeg(trainHamNames, testHamNames, trainSpamNames, testSpamNames,
                                           trainSpamDictionary, trainHamDictionary,
                                           allWordsCountTrainHam, allWordsCountTrainSpam,
                                           allWordsCountTestHam, allWordsCountTestSpam,
                                           stopWords, pSpam, pHam, alfa, V)
        # additive smoothing
        alfa = 0.01
        print()
        print("With additive smoothing factor: ", alfa)
        classificationErrorRateFalsePosNeg(trainHamNames, testHamNames, trainSpamNames, testSpamNames,
                                           trainSpamDictionary, trainHamDictionary,
                                           allWordsCountTrainHam, allWordsCountTrainSpam,
                                           allWordsCountTestHam, allWordsCountTestSpam,
                                           stopWords, pSpam, pHam, alfa, V)

    print('\n=======================================================')
    alfa = float(input('Additive smoothing factor (recommended: 0, 0.1, 0.2): '))
    print('Input to exit loop: "exit"')
    while True:
        emailName = input('Email name (extension included): ')
        if emailName == 'exit':
            break
        prediction = predictData(emailName, stopWords, pSpam, pHam,
                                 trainSpamDictionary, trainHamDictionary,
                                 allWordsCountTrainSpam, allWordsCountTrainHam, alfa, V)

        if prediction:
            print(f'{emailName} is SPAM')
        else:
            print(f'{emailName} is HAM')


if __name__ == '__main__':
    main()
