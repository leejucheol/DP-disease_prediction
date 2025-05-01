import React, { useState, useEffect } from "react";
import "./MainPage.css";
import Header from "../components/Header";
import WorkArea from "../components/WorkArea";
import WorkCategory from "../components/WorkCategory";

const MainPage: React.FC = () => {
    const [placeholder, setPlaceholder] = useState<string>("단백질 ID 혹은 이름을 입력해주세요...");

    useEffect(() => {
        // 기본값으로 단백질 검색 활성화
        setPlaceholder("단백질 ID 혹은 이름을 입력해주세요...");
    }, []);

    const handleCategorySelect = (id: string) => {
        if (id === "protein-search") {
            setPlaceholder("단백질 ID 혹은 이름을 입력해주세요...");
        } else if (id === "disease-search") {
            setPlaceholder("질병 ID 혹은 이름을 입력해주세요...");
        }
    };

    return (
        <div className="main-page">
            <WorkCategory onSelect={handleCategorySelect} defaultSelected="protein-search" />
            <div className="main-layout">
                <Header placeholder={placeholder} />
                <WorkArea />
            </div>
        </div>
    );
};

export default MainPage;
