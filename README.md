# Neural-ARM: Artificial Neural Network Approach for Accelerated Association Rule Mining

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Research Paper](https://img.shields.io/badge/Research-Published-green.svg)](https://www.ijraset.com/research-paper/association-rule-mining-using-fp-growth)

## 🎯 Project Overview

**Neural-ARM** presents an innovative approach to Association Rule Mining (ARM) using Artificial Neural Networks with Denoising Algorithms. This research project demonstrates significant performance improvements in execution time compared to traditional ARM algorithms like Apriori and FP-Growth, while maintaining high-quality association rules.

### 🔬 Research Significance
This project explores the application of **Deep Learning techniques** in the field of **Data Mining**, specifically using **denoising autoencoders** to accelerate the association rule mining process on transactional datasets.

## 📊 Key Features

- **🚀 Performance Enhancement**: Reduced execution time compared to traditional ARM algorithms
- **🧠 Neural Network Architecture**: Implementation using denoising autoencoders 
- **📈 Comparative Analysis**: Benchmarked against Apriori and FP-Growth algorithms
- **📋 Multiple Datasets**: Tested on grocery store and shop transaction datasets
- **📊 Comprehensive Metrics**: Support, confidence, lift, and interestingness measures
- **📝 Academic Validation**: Results published in peer-reviewed research papers

## 🏗️ Project Structure

```
Neural-ARM/
├── 📁 Model/                          # Core implementation
│   ├── Algorithm-using-ANN.ipynb     # Main ANN-based ARM implementation
│   ├── Apriori_datamining.ipynb      # Traditional Apriori algorithm
│   ├── FPGrowth.ipynb                # FP-Growth algorithm implementation
│   ├── main_prog.py                  # Main program entry point
│   └── dataset/                      # Transaction datasets
│       ├── groceries.csv
│       ├── shop-transaction-data.csv
│       └── shopdata.csv
├── 📁 Docs/                          # Research documentation
│   ├── Final_Report.pdf             # Complete research report
│   └── RP_9788770227667C9.pdf       # Published paper (River Publishers)
├── 📁 Images/                        # Visualizations and diagrams
│   ├── autoencoder.png
│   ├── denoising autoencoder.png
│   └── frequent_items_*.png
└── 📄 README.md                      # Project documentation
```

## 🔬 Methodology

### Traditional Approach vs Neural Approach

| Aspect | Traditional (Apriori/FP-Growth) | Neural-ARM (Our Approach) |
|--------|--------------------------------|---------------------------|
| **Algorithm Type** | Frequent Pattern Mining | Neural Network with Denoising |
| **Data Structure** | Tree-based/Candidate Generation | One-hot Encoded Vectors |
| **Processing** | Multiple Database Scans | Single Pass with Autoencoder |
| **Time Complexity** | O(n²) - O(n³) | Significantly Reduced |
| **Memory Usage** | High for Large Datasets | Optimized through Compression |

### 🧠 Neural Network Architecture

1. **Input Layer**: One-hot encoded transaction vectors
2. **Encoder**: Compresses transaction data into lower dimensions
3. **Denoising Component**: Reduces noise and enhances patterns
4. **Decoder**: Reconstructs transaction patterns
5. **Association Rule Extraction**: Derives rules from learned representations

## 📈 Performance Results

### Execution Time Comparison
- **FP-Growth Algorithm**: Higher execution time
- **Neural-ARM**: **Significantly reduced execution time**
- **Performance Gain**: Substantial improvement in processing speed

### Rule Quality Analysis
- **Support & Confidence**: Maintained high-quality thresholds
- **Lift Values**: Generated meaningful associations
- **Interestingness**: Focused on potentially strong rules only

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy pandas tensorflow scikit-learn matplotlib seaborn
```

### Running the Project
1. **Clone the repository**
   ```bash
   git clone https://github.com/pankaj-creator/final_year_project.git
   cd final_year_project
   ```

2. **Run the Neural-ARM Algorithm**
   ```bash
   cd Model
   jupyter notebook Algorithm-using-ANN.ipynb
   ```

3. **Compare with Traditional Algorithms**
   ```bash
   jupyter notebook Apriori_datamining.ipynb
   jupyter notebook FPGrowth.ipynb
   ```

## 📚 Research Publications

This research has been validated through peer-reviewed publications:

### 📄 Published Papers

1. **[Association Rule Mining using FP-Growth and An Innovative Artificial Neural Network Techniques](https://www.ijraset.com/research-paper/association-rule-mining-using-fp-growth)**
   - **Journal**: International Journal for Research in Applied Science and Engineering Technology (IJRASET)
   - **DOI**: [10.22214/ijraset.2022.43149](https://doi.org/10.22214/ijraset.2022.43149)
   - **Authors**: Pankaj Kumar Gond, Aditya Shukla, Satish Sahani, Neha Gond, Dr. Harvendra Kumar
   - **Publication Date**: May 2022

2. **[River Publishers Chapter](https://www.riverpublishers.com/pdf/ebook/chapter/RP_9788770227667C9.pdf)**
   - **Publisher**: River Publishers
   - **Chapter**: Advanced Techniques in Association Rule Mining
   - **Status**: Published

### 📑 Complete Documentation
- **[📋 Final Research Report](./Docs/Final_Report.pdf)**: Comprehensive analysis and results
- **[📊 Research Paper Draft](./Docs/R_paper(V2).docx)**: Latest version with detailed methodology

## 🔧 Technical Implementation

### Core Technologies
- **Python 3.6+**: Primary programming language
- **TensorFlow/Keras**: Neural network implementation
- **NumPy & Pandas**: Data processing and analysis
- **Scikit-learn**: Machine learning utilities
- **Matplotlib & Seaborn**: Data visualization

### Key Algorithms Implemented
1. **Neural-ARM**: Novel ANN-based association rule mining
2. **Traditional Apriori**: Baseline comparison algorithm
3. **FP-Growth**: Tree-based frequent pattern mining
4. **Performance Metrics**: Support, confidence, lift calculation

## 📊 Datasets

### Transaction Datasets Used
1. **Groceries Dataset**: Retail transaction data
2. **Shop Transaction Data**: General merchandise transactions
3. **Custom Datasets**: Formatted for comparative analysis

### Data Preprocessing
- **Data Cleaning**: Removal of null values and inconsistencies
- **One-hot Encoding**: Conversion to neural network compatible format
- **Normalization**: Feature scaling for optimal training

## 🎯 Key Contributions

1. **Algorithmic Innovation**: Novel application of denoising autoencoders in ARM
2. **Performance Improvement**: Significant reduction in execution time
3. **Quality Maintenance**: Preserved rule quality while improving speed
4. **Comparative Analysis**: Comprehensive evaluation against existing methods
5. **Academic Impact**: Peer-reviewed publication and research validation

## 👥 Research Team

- **Pankaj Kumar Gond** - Lead Researcher & Developer
- **Aditya Shukla** - Co-Researcher
- **Dr. Harvendra Kumar** - Research Supervisor
- **Satish Sahani** - Technical Contributor  
- **Neha Gond** - Research Assistant

## 🏆 Awards & Recognition

- **Research Publication**: Successfully published in international journals
- **Academic Excellence**: Final year project with innovative approach
- **Technical Innovation**: Novel application of neural networks in data mining

## 🔮 Future Work

- **Scalability Enhancement**: Optimization for larger datasets
- **Real-time Processing**: Implementation for streaming data
- **Deep Learning Integration**: Advanced neural architectures
- **Cloud Deployment**: Distributed computing implementation

## 📞 Contact & Collaboration

**Primary Contact**: Pankaj Kumar Gond  
**Email**: [Contact via GitHub Issues](https://github.com/pankaj-creator/final_year_project/issues)  
**Research Supervisor**: Dr. Harvendra Kumar  

For research collaboration or technical inquiries, please open an issue or contact the research team.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Research Institution**: For providing computational resources
- **Academic Supervisors**: For guidance and mentorship  
- **Peer Reviewers**: For valuable feedback and validation
- **Open Source Community**: For tools and libraries used

---

**⭐ Star this repository if you find this research useful!**

**📚 [Read our published paper 1](https://www.ijraset.com/research-paper/association-rule-mining-using-fp-growth) • 📚 [Read our published paper 2](https://www.riverpublishers.com/pdf/ebook/chapter/RP_9788770227667C9.pdf) • 📊 [View complete report](./Docs/Final_Report.pdf) • 🔬 [Explore the code](./Model/)**

