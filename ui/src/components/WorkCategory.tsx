import React, { useState, useEffect } from "react";
import "./WorkCategory.css";
import { useNavigate } from "react-router-dom";

interface WorkCategoryProps {
    onSelect: (id: string) => void;
    defaultSelected?: string; // 기본 선택값 추가
}

const WorkCategory: React.FC<WorkCategoryProps> = ({ onSelect, defaultSelected }) => {
    const navigate = useNavigate();
    const [selected, setSelected] = useState<string | null>(null);

    useEffect(() => {
        // 기본 선택값 설정
        if (defaultSelected) {
            setSelected(defaultSelected);
            onSelect(defaultSelected);
        }
    }, [defaultSelected, onSelect]);

    const handleSelect = (id: string) => {
        setSelected(id);
        onSelect(id);
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
