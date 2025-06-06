// - Xiao Gao - xgao045 
// - DatasetID: small_data_39, large_data_31

// - Small Dataset Results:
// - Forward: Feature Subset: <1,10>, Acc: 96.2%
// - Backward: Feature Subset: <4,7, 10>, Acc: 86.4%

// - Large Dataset Results:
// - Forward: Feature Subset: <32, 44>, Acc: 97.1%
// - Backward: Feature Subset: <3,5,6,9,11,12,13,14,17,22,23,24,25,26,29,31,32,34,35,36,37,41,45,46,47,48,49,50,51>, Acc: 77.7%

// - Iranian Churn Dataset Results:
// - Forward: Feature Subset: <1,2,3,4,5,6,7>, Acc: 93.4%
// - Backward: Feature Subset: <>, Acc: 78.0112%


#include <iostream>
#include <vector>
#include <set>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <ctime>
#include <iomanip>
#include <limits>
#include <numeric>
#include <locale>
#include <codecvt>
using namespace std;

// ----- CLASSIFIER CLASSES -----

class NearestNeighborClassifier {
public:
    NearestNeighborClassifier(const vector<int>& FeatureSet) : FeatureSet(FeatureSet) {}

    void train(const vector<vector<double>>& data, const vector<int>& labels) {
        TrainData = data;
        TrainLabels = labels;
    }

    int test(const vector<double>& TestPoint) const {
        double MinDistance = numeric_limits<double>::max();
        int NearestLabel = -1;

        for (size_t i = 0; i < TrainData.size(); ++i) {
            double distance = EuclideanDistance(TrainData[i], TestPoint);
            if (distance < MinDistance) {
                MinDistance = distance;
                NearestLabel = TrainLabels[i];
            }
        }

        return NearestLabel;
    }

private:
    vector<vector<double>> TrainData;
    vector<int> TrainLabels;
    vector<int> FeatureSet;

    double EuclideanDistance(const vector<double>& a, const vector<double>& b) const {
        double sum = 0.0;
        for (int index : FeatureSet) {
            sum += (a[index] - b[index]) * (a[index] - b[index]);
        }
        return sqrt(sum);
    }
};

class LeaveOneOutValidator {
public:
    LeaveOneOutValidator(const vector<int>& FeatureSet) : FeatureSet(FeatureSet) {}

    double validate(const vector<vector<double>>& data, const vector<int>& labels) {
        int CorrectPredictions = 0;
        NearestNeighborClassifier classifier(FeatureSet);

        for (size_t i = 0; i < data.size(); ++i) {
            vector<vector<double>> TrainData;
            vector<int> TrainLabels;

            for (size_t j = 0; j < data.size(); ++j) {
                if (j != i) {
                    TrainData.push_back(data[j]);
                    TrainLabels.push_back(labels[j]);
                }
            }

            classifier.train(TrainData, TrainLabels);
            int PredictedLabel = classifier.test(data[i]);

            if (PredictedLabel == labels[i]) {
                ++CorrectPredictions;
            }
        }

        return static_cast<double>(CorrectPredictions) / data.size();
    }

private:
    vector<int> FeatureSet;
};

// ----- FEATURE SELECTION CLASS -----

class FeatureSelection {
public:
    FeatureSelection(int totalFeatures, const vector<vector<double>>& data, const vector<int>& labels)
        : totalFeatures(totalFeatures), data(data), labels(labels) {}

    void forwardSelection() {
        set<int> currentFeatures;
        set<int> bestFeatures;
        double bestAccuracy = 0.0;


        cout << "\nUsing no features and leave-one-out evaluation, accuracy is "
             << leaveOneOutEvaluation(currentFeatures) * 100.0 << "%\n\n";

        cout << "Beginning search.\n\n";

        for (int i = 0; i < totalFeatures; ++i) {
            int bestFeatureToAdd = -1;
            double iterationBestAccuracy = 0.0;

            for (int feature = 0; feature < totalFeatures; ++feature) {
                if (currentFeatures.find(feature) == currentFeatures.end()) {
                    set<int> tempFeatures = currentFeatures;
                    tempFeatures.insert(feature);

                    double accuracy = leaveOneOutEvaluation(tempFeatures);
                    cout << "Using feature(s) {";
                    printSet(tempFeatures);
                    cout << "} accuracy is " << accuracy * 100.0 << "%\n";

                    if (accuracy > iterationBestAccuracy) {
                        iterationBestAccuracy = accuracy;
                        bestFeatureToAdd = feature;
                    }
                }
            }

            if (bestFeatureToAdd != -1) {
                currentFeatures.insert(bestFeatureToAdd);
                cout << "Feature set {";
                printSet(currentFeatures);
                cout << "} was best, accuracy is " << iterationBestAccuracy * 100.0 << "%\n\n";

                if (iterationBestAccuracy > bestAccuracy) {
                    bestAccuracy = iterationBestAccuracy;
                    bestFeatures = currentFeatures;
                }
            } else {
                cout << "No improvement found. Stopping search.\n";
                break;
            }
        }

        cout << "Finished search. Best feature subset is {";
        printSet(bestFeatures);
        cout << "} with accuracy of " << bestAccuracy * 100.0 << "%\n";
    }

    void backwardElimination() {
        set<int> currentFeatures;
        for (int feature = 1; feature <= totalFeatures; ++feature) {
            currentFeatures.insert(feature);
        }

        set<int> bestFeatures = currentFeatures;
        double bestAccuracy = leaveOneOutEvaluation(currentFeatures);
        cout << "Using all features and leave-one-out evaluation, accuracy is "
             << bestAccuracy * 100.0 << "%\n\n";

        while (currentFeatures.size() > 1) {
            int worstFeatureToRemove = -1;
            double iterationBestAccuracy = 0.0;

            for (int feature : currentFeatures) {
                set<int> tempFeatures = currentFeatures;
                tempFeatures.erase(feature);

                double accuracy = leaveOneOutEvaluation(tempFeatures);
                cout << "Using feature(s) {";
                printSet(tempFeatures);
                cout << "} accuracy is " << accuracy * 100.0 << "%\n";

                if (accuracy > iterationBestAccuracy) {
                    iterationBestAccuracy = accuracy;
                    worstFeatureToRemove = feature;
                }
            }

            if (worstFeatureToRemove != -1) {
                currentFeatures.erase(worstFeatureToRemove);
                cout << "Feature set {";
                printSet(currentFeatures);
                cout << "} was best, accuracy is " << iterationBestAccuracy * 100.0 << "%\n\n";

                if (iterationBestAccuracy > bestAccuracy) {
                    bestAccuracy = iterationBestAccuracy;
                    bestFeatures = currentFeatures;
                }
            } else {
                cout << "No improvement found. Stopping search.\n";
                break;
            }
        }

        cout << "Finished search. Best feature subset is {";
        printSet(bestFeatures);
        cout << "} with accuracy of " << bestAccuracy * 100.0 << "%\n";
    }

private:
    int totalFeatures;
    vector<vector<double>> data;
    vector<int> labels;

    double leaveOneOutEvaluation(const set<int>& features) {
        vector<int> featureSet(features.begin(), features.end());
        LeaveOneOutValidator validator(featureSet);
        return validator.validate(data, labels);
    }

    void printSet(const set<int>& s) const {
        for (auto it = s.begin(); it != s.end(); ++it) {
            if (it != s.begin()) cout << ",";
            cout << *it + 1;
        }
    }
};

// ----- DATA LOADING FUNCTION -----

bool LoadData(const string& filename, vector<vector<double>>& data, vector<int>& labels) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return false;
    }

    string line;
    while (getline(file, line)) {
        vector<double> features;
        int label;
        stringstream ss(line);

        ss >> label;
        labels.push_back(label);
        
        double value;
        while (ss >> value) {
            features.push_back(value);
        }

        features.erase(features.begin());
        data.push_back(features);
        
    }

    file.close();
    return true;
}

bool RealData(const string& filename, vector<vector<double>>& data, vector<int>& labels){
    wifstream wif(filename, ios::binary);
    if (!wif.is_open()) return false;
    // Set UTF-16 locale
    wif.imbue(locale(wif.getloc(), new codecvt_utf16<wchar_t, 0x10ffff, std::little_endian>()));
    wchar_t bom;
    wif.read(&bom, 1);
    if (bom != 0xFEFF) wif.seekg(0);

    wstring line;
    while (getline(wif, line)) {
        if (line.empty()) continue;
        wstringstream wss(line);
        vector<double> row;
        wstring token;
        bool isFirst = true;
        int lbl = 0;
        while (getline(wss, token, L'\t')) {
            // Remove nulls
            wstring clean;
            for (auto c : token) if (c != L'\0') clean += c;
            if (isFirst) {
                lbl = stoi(clean);
                isFirst = false;
            } else {
                if (!clean.empty()) row.push_back(stod(clean));
            }
        }
        if (!row.empty()) {
            labels.push_back(lbl);
            data.push_back(row);
        }
    }
    return true;

}



// ----- MAIN FUNCTION -----

int main() {
    vector<vector<double>> data;
    vector<int> labels;
    clock_t start, end; // timer for overall execution

    cout << "Welcome to Xiao's Feature Selection Algorithm.\n";

    cout << "Select dataset:\n1. Small Dataset (12 features)\n2. Large Dataset (50 features)\n3. Iranian Churn Dataset (8 features)\n";
    int choice;
    cin >> choice;

    string filepath;
    vector<int> FeatureSet; // initialization
    if (choice == 1) {
        filepath = "CS205_small_Data__39.txt"; // load the small data
        if (!LoadData(filepath, data, labels)) {
        cerr << "Failed to load dataset.\n";
        return 1;
        }
    }
    else if (choice == 2) {
        filepath = "CS205_large_Data__31.txt"; // load the large data
        if (!LoadData(filepath, data, labels)) {
        cerr << "Failed to load dataset.\n";
        return 1;
        }
    }
    else if (choice == 3) {
        filepath = "Customer_Churn.txt"; // load the real data
        if (!RealData(filepath, data, labels)) {
        cerr << "Failed to load dataset.\n";
        return 1;
        }
    }
    else {
        cerr << "Invalid choice. Exiting program.\n";
        return 1;
    }

    start = clock(); // start overall timer


    // Get the count of all instances
    int totalInstances = data.size(); // The number of rows in the dataset

    // Get the count of all features (excluding the class attribute)
    int totalFeatures = data[0].size(); // The total number of columns in the dataset

    cout << endl << "This dataset has " << totalFeatures 
        << " features (not including the class attribute), with " 
        << totalInstances << " instances.\n";


    // Normalize data
    for (size_t j = 0; j < data[0].size(); ++j) {
        double min = numeric_limits<double>::max();
        double max = numeric_limits<double>::lowest();
        for (size_t i = 0; i < data.size(); ++i) {
            min = min < data[i][j] ? min : data[i][j];
            max = max > data[i][j] ? max : data[i][j];
        }
        for (size_t i = 0; i < data.size(); ++i) {
            data[i][j] = (data[i][j] - min) / (max - min);
        }
    }

    // int totalFeatures = data[0].size();
    // cout << "Total Feature: " << totalFeatures;
    // for (int i = 0; i < totalFeatures; ++i) {
    //     cout << data[0][i] << " ";
    // }

    FeatureSelection fs(totalFeatures, data, labels);

    cout << "\nType the number of the algorithm you want to run.\n";
    cout << "1. Forward Selection" << endl;
    cout << "2. Backward Elimination" << endl;
    cin >> choice;

    if (choice == 1) {
        fs.forwardSelection();
    }
    else if (choice == 2) {
        fs.backwardElimination();
    }
    else {
        cout << "Invalid choice. Please select 1 or 2.\n";
    }

    // end overall time
    end = clock(); 
    double duration = (double (end - start)) / CLOCKS_PER_SEC;
    cout << "Total time: " << duration << "seconds" << endl;


    return 0;
}