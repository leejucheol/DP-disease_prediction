import React, { useState } from "react";
import "./Header.css";

interface HeaderProps {
    placeholder: string;
    onSearch: (searchText: string) => void;
}

const Header: React.FC<HeaderProps> = ({ placeholder, onSearch }) => {
    const [showModal, setShowModal] = useState(false);
    const [searchText, setSearchText] = useState("");

    const handleSearch = () => {
        if (searchText.trim()) {
            onSearch(searchText);
        } else {
            alert("검색어를 입력해주세요!");
        }
    };

    const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === "Enter") {
            handleSearch();
        }
    };

    return (
        <>
            <header className="header">
                <div className="logo-area">
                    <p className="logo">Disease Prediction</p>
                </div>
                <div className="search-area">
                    <button className="search-button" onClick={handleSearch}>
                        검색
                    </button>
                    <input
                        type="text"
                        className="search-input"
                        placeholder={placeholder}
                        value={searchText}
                        onChange={(e) => setSearchText(e.target.value)}
                        onKeyDown={handleKeyPress}
                    />
                </div>
                <div className="signin-area">
                    <div className="signin-button" onClick={() => setShowModal(true)}>
                        <p>Sign in</p>
                    </div>
                </div>
            </header>

            {/* 모달 */}
            {showModal && (
                <div className="modal-overlay">
                    <div className="modal-content">
                        <h3>로그인</h3>
                        <button className="google-login-button">Google 계정으로 시작하기</button>
                        <button className="close-button" onClick={() => setShowModal(false)}>
                            닫기
                        </button>
                    </div>
                </div>
            )}
        </>
    );
};

export default Header;
