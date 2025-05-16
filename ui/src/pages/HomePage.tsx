import React, { useEffect, useState } from "react";
import "./HomePage.css";
import { useNavigate } from "react-router-dom";

const HomePage: React.FC = () => {
    const navigate = useNavigate();
    const [text, setText] = useState("");
    const fullText =
        "사용자는 단백질 서열을 입력하고, DP는 해당 단백질과 관련된 질병을 예측합니다. 또한 질병을 입력하면 관련 단백질을 예측합니다.";

    useEffect(() => {
        const timeout = setTimeout(() => {
            let index = 0;
            const interval = setInterval(() => {
                setText(fullText.slice(0, index)); // 현재 인덱스까지 슬라이싱
                index++;

                if (index > fullText.length) {
                    clearInterval(interval);
                }
            }, 30); // 글자 출력 속도 조절 (30ms)

            return () => clearInterval(interval);
        }, 500); // 0.5초 뒤에 실행

        return () => clearTimeout(timeout);
    }, []);

    return (
        <div className="home-page">
            <div
                className="main-banner"
                style={{
                    width: "100vw",
                    height: "100vh",
                    backgroundImage: `url(${process.env.PUBLIC_URL}/mainLayout.png)`,
                    backgroundSize: "cover",
                    backgroundPosition: "center",
                    backgroundRepeat: "no-repeat",
                    display: "flex",
                    justifyContent: "center",
                    alignItems: "center",
                    position: "relative",
                }}
            >
                <div className="content">
                    <p className="title">Disease Prediction</p>

                    <button className="start-button" onClick={() => navigate("/")}>
                        시작하기
                    </button>
                </div>

                {/* 왼쪽 하단 설명 div */}
                <div className="description">
                    <p className="description-title">
                        DP은 단백질과 질병의 관계를 분석하여 질병 또는 연관 단백질을 예측합니다.
                    </p>
                    <span className="typing-effect">{text}</span>
                </div>
            </div>
        </div>
    );
};

export default HomePage;
