import React from "react";
import "./WorkArea.css";

interface Prediction {
    disease_id: string;
    disease_name: string;
    probability: number;
}

interface PredictionResult {
    sequence: string;
    predictions: Prediction[];
}

interface WorkAreaProps {
    predictionResult: PredictionResult | null;
    loading: boolean;
}

const WorkArea: React.FC<WorkAreaProps> = ({ predictionResult, loading }) => {
    if (loading) {
        return (
            <div className="work-area">
                <div className="loading-container">
                    <div className="loading-spinner"></div>
                    <p>분석 중입니다...</p>
                </div>
            </div>
        );
    }

    if (!predictionResult) {
        return (
            <div className="work-area">
                <div className="empty-state">
                    <p>단백질 시퀀스를 입력하여 질병 예측 결과를 확인하세요.</p>
                </div>
            </div>
        );
    }

    return (
        <div className="work-area">
            <div className="result-container">
                <h2>예측 결과</h2>
                <div className="sequence-info">
                    <h3>입력된 시퀀스</h3>
                    <div className="sequence-box">
                        {predictionResult.sequence.length > 100
                            ? `${predictionResult.sequence.substring(0, 100)}...`
                            : predictionResult.sequence}
                        <span className="sequence-length">총 {predictionResult.sequence.length}개 아미노산</span>
                    </div>
                </div>

                <div className="predictions-container">
                    <h3>예측된 질병 (가능성 내림차순)</h3>
                    <div className="predictions-list">
                        {predictionResult.predictions.map((prediction, index) => (
                            <div className="prediction-item" key={prediction.disease_id}>
                                <div className="prediction-rank">{index + 1}</div>
                                <div className="prediction-info">
                                    <h4>{prediction.disease_name}</h4>
                                    <p className="disease-id">ID: {prediction.disease_id}</p>
                                </div>
                                <div className="prediction-probability">
                                    <div className="probability-bar">
                                        <div
                                            className="probability-fill"
                                            style={{ width: `${prediction.probability * 100}%` }}
                                        ></div>
                                    </div>
                                    <p>{(prediction.probability * 100).toFixed(2)}%</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default WorkArea;
