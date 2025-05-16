import React, { useState, useEffect } from "react";
import "./WorkCategory.css";
import { useNavigate } from "react-router-dom";

interface WorkCategoryProps {
    onSelect: (id: string) => void;
    defaultSelected?: string; // 기본 선택값 추가
}

const WorkCategory: React.FC<WorkCategoryProps> = ({ onSelect, defaultSelected }) => {
    const navigate = useNavigate();
    // 초기 상태만 defaultSelected로 설정하고, 이후에는 사용자 선택에 따라 변경
    const [selected, setSelected] = useState<string>(defaultSelected || "protein-search");

    // 초기 렌더링시에만 defaultSelected 적용
    useEffect(() => {
        if (defaultSelected) {
            setSelected(defaultSelected);
            onSelect(defaultSelected);
        }
    }, []); // 의존성 배열을 비워서 컴포넌트 마운트 시에만 실행

    const handleSelect = (id: string) => {
        setSelected(id);
        onSelect(id);
        console.log("Category selected:", id); // 디버깅 로그
    };

    return (
        <div id="work-category">
            <div id="intro-dp" onClick={() => navigate("/intro")}>
                <p id="go-info-btn">DP information</p>
            </div>
            <div
                id="protein-search-btn"
                className={`search-select-btn ${selected === "protein-search" ? "active" : ""}`}
                onClick={() => handleSelect("protein-search")}
            >
                <p>단백질 검색</p>
            </div>
            <div
                id="disease-search-btn"
                className={`search-select-btn ${selected === "disease-search" ? "active" : ""}`}
                onClick={() => handleSelect("disease-search")}
            >
                <p>질병 검색</p>
            </div>
        </div>
    );
};

export default WorkCategory;
