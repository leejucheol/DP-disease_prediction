import React, { useState, useEffect } from "react";
import "./MainPage.css";
import Header from "../components/Header";
import WorkArea from "../components/WorkArea";
import WorkCategory from "../components/WorkCategory";

interface Prediction {
    disease_id: string;
    disease_name: string;
    probability: number;
}

interface PredictionResult {
    sequence: string;
    predictions: Prediction[];
}

const MainPage: React.FC = () => {
    const [placeholder, setPlaceholder] = useState<string>("단백질 ID 혹은 이름을 입력해주세요...");
    const [selectedCategory, setSelectedCategory] = useState<string>("protein-search");
    const [loading, setLoading] = useState<boolean>(false);
    const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);

    useEffect(() => {
        // 기본값으로 단백질 검색 활성화
        setPlaceholder("단백질 ID 혹은 이름을 입력해주세요...");
    }, []);

    const handleCategorySelect = (id: string) => {
        setSelectedCategory(id);
        if (id === "protein-search") {
            setPlaceholder("단백질 ID 혹은 이름을 입력해주세요...");
        } else if (id === "disease-search") {
            setPlaceholder("질병 ID 혹은 이름을 입력해주세요...");
            // 카테고리 변경 시 결과 초기화
            setPredictionResult(null);
        }
    };

    const handleSearch = async (searchText: string) => {
        if (selectedCategory === "protein-search") {
            setLoading(true); // 로딩 시작
            console.log("검색 시작, 로딩 상태:", loading);

            try {
                console.log("API 요청 시작:", searchText);
                const response = await fetch("http://localhost:8000/proteins/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({ sequence: searchText }),
                });

                console.log("API 응답 받음:", response.status);

                if (!response.ok) {
                    throw new Error(`API 요청 실패: ${response.status}`);
                }

                const data = await response.json();
                console.log("응답 데이터:", data);
                setPredictionResult(data);
            } catch (error) {
                console.error("API 요청 오류:", error);
                alert(`예측 중 오류가 발생했습니다: ${error instanceof Error ? error.message : "알 수 없는 오류"}`);
            } finally {
                // 성공하든 실패하든 로딩 상태를 false로 설정
                console.log("로딩 상태 해제");
                setLoading(false);
            }
        } else if (selectedCategory === "disease-search") {
            // 질병 검색 기능 구현 (아직 미구현)
            alert("질병 검색 기능은 아직 개발 중입니다.");
        }
    };

    // 디버깅용 로그 추가
    useEffect(() => {
        console.log("로딩 상태 변경:", loading);
    }, [loading]);

    useEffect(() => {
        console.log("예측 결과 변경:", predictionResult);
    }, [predictionResult]);

    return (
        <div className="main-page">
            <WorkCategory onSelect={handleCategorySelect} defaultSelected="protein-search" />
            <div className="main-layout">
                <Header placeholder={placeholder} onSearch={handleSearch} />
                <WorkArea predictionResult={predictionResult} loading={loading} />
            </div>
        </div>
    );
};

export default MainPage;
