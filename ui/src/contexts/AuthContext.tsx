import React, { ReactNode, createContext, useEffect, useState } from "react";

interface AuthContextType {
    userId: number | null;
    avatarId: number | null;
    nickname: string | null;
    isLoggedIn: boolean;
    setUserId: (userId: number | null) => void;
    setNickname: (nickname: string | null) => void;
    setAvatarId: (avatarId: number | null) => void;
    login: (userId: number, avatarId: number, nickname: string, jwt: string) => void;
    logout: () => void;
}

export const AuthContext = createContext<AuthContextType>({
    userId: null,
    avatarId: null,
    nickname: null,
    isLoggedIn: false,
    setUserId: () => {},
    setAvatarId: () => {},
    setNickname: () => {},
    login: () => {},
    logout: () => {},
});

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
    const [userId, setUserId] = useState<number | null>(null);
    const [avatarId, setAvatarId] = useState<number | null>(null);
    const [nickname, setNickname] = useState<string | null>(null);
    const [isLoggedIn, setIsLoggedIn] = useState<boolean>(false);

    useEffect(() => {
        // 세션에서 jwt토큰과 userId 가져옴
        const token = sessionStorage.getItem("jwt");
        const storedUserId = sessionStorage.getItem("userId");
        const storedAvatarId = sessionStorage.getItem("avatarId");

        // 토큰과 사용자 id가 있을 경우 로그인 상태를 유지
        if (token && storedUserId) {
            setIsLoggedIn(true);
            setUserId(Number(storedUserId));
            setAvatarId(Number(storedAvatarId));
        } else {
            setIsLoggedIn(false);
        }
    }, []);

    // 로그인 버튼에 필요한 로그인 로직
    const login = (userId: number, avatarId: number, nickname: string, jwt: string) => {
        // 세션 스토리지에 jwt와 userId 저장
        sessionStorage.setItem("jwt", jwt);
        sessionStorage.setItem("userId", userId.toString());
        sessionStorage.setItem("avatarId", avatarId.toString());
        setNickname(nickname);
        setUserId(userId);
        setAvatarId(avatarId);
        setIsLoggedIn(true);
    };

    // 로그아웃 버튼에 필요한 로그아웃 로직
    const logout = () => {
        // 로그인 상태, 사용자 id, 닉네임 비움
        setUserId(null);
        setAvatarId(null);
        setNickname(null);
        setIsLoggedIn(false);
        // 세션 스토리지에서 jwt와 userId 삭제
        sessionStorage.removeItem("jwt");
        sessionStorage.removeItem("userId");
        sessionStorage.removeItem("avatarId");
    };

    return (
        <AuthContext.Provider
            value={{
                userId,
                avatarId,
                nickname,
                isLoggedIn,
                setUserId,
                setAvatarId,
                setNickname,
                login,
                logout,
            }}
        >
            {children}
        </AuthContext.Provider>
    );
};
