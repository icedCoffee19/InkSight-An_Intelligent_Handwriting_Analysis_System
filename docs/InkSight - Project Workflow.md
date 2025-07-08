

**InkSight - Project Workflow**



Comprehensive workflow and design plan with tech stack suggestions, data sources, and model 

architecture decisions that align with your objectives.

&nbsp;

**1. Finalized Project Workflow for InkSight**

&nbsp;	**A. User Interface (Frontend Web App)**

&nbsp;	\* Upload handwriting image

&nbsp;	\* Choose task:

&nbsp;		o Handwriting Recognition (HCR)

&nbsp;		o Personality Profiling via Graphology

&nbsp;		o Both

&nbsp;	\* Show dashboard: recognized text + personality profile charts/graphs

&nbsp;

&nbsp;	**B. Backend (AI + Processing Engine)**

&nbsp;	\* Preprocessing: noise removal, skew correction, binarization

&nbsp;	\* Module 1 - HCR Pipeline:

&nbsp;		o CNN or CNN-RNN model for character/word recognition

&nbsp;		o Return transcribed text

&nbsp;	\* Module 2 - Graphology Engine:

&nbsp;		o Feature extraction from handwriting (slant, pressure, spacing, loops)

&nbsp;		o Predict psychological traits (MBTI/Big Five/stress/cognitive health)

&nbsp;		o Display trait confidence scores with visualization

&nbsp;	\* Result Storage:

&nbsp;		o MongoDB/Firestore for storing user input, output, and feedback

&nbsp;

	**C. Output Visualization Dashboard**

&nbsp;	\* BI visualizations (using libraries like Chart.js, D3.js, or Plotly)

&nbsp;	\* Graphs: slant chart, pressure heatmap, personality radar/spider graph

&nbsp;	\* Text outputs + downloadable results

&nbsp;

**2. Finalized Technology Stack**

&nbsp;	**Component		Recommended Tools (Beginner Friendly)**

&nbsp;	Frontend		HTML/CSS + JavaScript + React.js (for interactivity)

&nbsp;	UI Components		Tailwind CSS or Bootstrap + ShadCN (for fast UI building)

&nbsp;	Backend API		FastAPI (Python) - beginner-friendly, fast

&nbsp;	Model Serving		TensorFlow/Keras or PyTorch + ONNX for exporting models

&nbsp;	Database		MongoDB (document-based, good for flexible JSON-style storage)

&nbsp;	Image Uploads		Firebase Storage or Cloudinary

&nbsp;	Deployment		Vercel (Frontend) + Render or Railway (for Python backend)

&nbsp;	Dashboard Tools		Plotly.js, Chart.js, or Streamlit (if dashboard is backend-generated)

&nbsp;	BI/Analytics		Optional: Integrate with Power BI/Google Data Studio using APIs or connectors

&nbsp;

**3. Recommended Datasets**

&nbsp;	**Module 		Dataset**

&nbsp;	HCR		IAM Dataset, CVL, EMNIST (you already have these)

&nbsp;	Graphology	IAM + Custom annotated dataset (MBTI traits, stress levels, etc.)

&nbsp;			Multilingual Expansion (future): KHATT (Arabic), Devanagari, Bangla handwriting sets

&nbsp;

**4. Suggested Models for Processing Modules**

	**HCR Module**

&nbsp;	**Stage			Model/Method**

&nbsp;	Preprocessing		OpenCV (binarization, denoising, skew correction)

&nbsp;	Feature Extraction	CNN (e.g., ResNet, VGG) on character/word images

&nbsp;	Sequence Learning 	CNN + BiLSTM + CTC (Connectionist Temporal Classification) for full lines

	**Output			Transcribed text in English (extendable to other languages)**

    **Use pretrained models on IAM or EMNIST initially, then fine-tune.**

&nbsp;

	**Graphology Module**

&nbsp;	Stage			Method

&nbsp;	Structural Features	Slant, loops, pressure, baseline, spacing - from image using OpenCV

&nbsp;	Feature Classifier	Use Random Forest / SVM / Gradient Boosting initially

&nbsp;	Deep Learning		CNN on whole image (e.g., personality classification)

&nbsp;	Hybrid Model		Combine handcrafted features + CNN embedding for best results

&nbsp;	Output			MBTI/Big Five traits + stress + mood + cognitive markers

    *Inspired by papers like Gagiu et al. (2025), Kim et al. (2022)*

&nbsp;

**5. Privacy \& Ethics Layer**

&nbsp;	**Concern					Solution**

&nbsp;	Identity Protection			Separate user identity from data using hash or UUID

&nbsp;	Data Minimization			Store only essential features and anonymized text

&nbsp;	Explainability \& Transparency		Show "Why" behind predictions (rules or saliency maps)

&nbsp;	Consent \& Opt-out			Provide checkbox for data usage/feedback during upload

&nbsp;

**6. Project Timeline Suggestion (Modular Milestones)**

&nbsp;	**Phase		Duration		Goal**

&nbsp;	Phase 1		Week 1-2		Frontend form + backend upload API + storage setup

&nbsp;	Phase 2		Week 3-4		Implement basic HCR model and display results

&nbsp;	Phase 3		Week 5-6		Add graphology engine with basic rule-based classification

&nbsp;	Phase 4		Week 7-8		Add dashboard with personality trait visualization

&nbsp;	Phase 5		Week 9-10		Train improved models + real-time feedback + final polish





