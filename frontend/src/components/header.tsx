import React, { useState, useRef, useEffect } from "react";

const Header: React.FC = () => {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [showSearch, setShowSearch] = useState(false);
  const mobileRef = useRef<HTMLDivElement | null>(null);
  const toggleRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      const target = e.target as Node;
      if (mobileRef.current && mobileRef.current.contains(target)) return;
      if (toggleRef.current && toggleRef.current.contains(target)) return;
      setMobileOpen(false);
    }
    document.addEventListener("click", handleClick);
    return () => document.removeEventListener("click", handleClick);
  }, []);

  return (
    <header className="w-full overflow-x-hidden">
      <div className="max-w-7xl mx-auto px-4  py-3">
        <div className="h-20 flex items-center justify-between gap-6">
          <div className="flex items-center gap-4 ml-[-20px]">
            <svg width={100} height={100} viewBox="0 0 370 370" fill="#ef3124">
              <rect x="114.28" y="258.75" width="141.44" height="29.39" />
              <path d="M210.89,94.41c-4.03-12.03-8.68-21.53-24.61-21.53s-20.87,9.46-25.12,21.53l-43.76,124.41h29.02l10.1-29.58h55.84l9.37,29.58h30.86l-41.71-124.41Zm-45.91,69.85l19.84-58.96h.73l18.74,58.96h-39.31Z" />
            </svg>
            <button
              ref={toggleRef}
              className="lg:hidden ml-1 p-2 rounded-md hover:bg-gray-100"
              aria-label="Открыть меню"
              onClick={(e) => {
                e.stopPropagation();
                setMobileOpen((s) => !s);
              }}
            >
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M4 6h16M4 12h16M4 18h16"
                  stroke="#111827"
                  strokeWidth="1.6"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
          </div>

          <nav className="hidden lg:flex flex-1 justify-center gap-12 text-lg font-medium min-w-0">
            <div className="flex gap-8">
              <a href="#" className="text-black/90 hover:text-black">
                Контакты
              </a>
              <a href="#" className="text-black/90 hover:text-black">
                Информация
              </a>
              <a href="#" className="text-black/90 hover:text-black">
                Подразделения
              </a>
            </div>
          </nav>

          <div className="flex items-center gap-3">
            <div className="hidden lg:block relative min-w-0">
              <span className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500 pointer-events-none">
                <svg
                  width="18"
                  height="18"
                  viewBox="0 0 24 24"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    d="M21 21l-4.35-4.35"
                    stroke="#6B7280"
                    strokeWidth="1.6"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <circle
                    cx="11"
                    cy="11"
                    r="5"
                    stroke="#6B7280"
                    strokeWidth="1.6"
                  />
                </svg>
              </span>
              <input
                aria-label="Поиск"
                placeholder="Поиск"
                className="pl-10 pr-4 py-2 rounded-full bg-gray-100 text-sm w-full max-w-xs focus:outline-none min-w-0"
              />
            </div>

            <button
              className="lg:hidden p-2 rounded-full hover:bg-gray-100"
              aria-label="Поиск"
              onClick={(e) => {
                e.stopPropagation();
                setShowSearch((s) => !s);
              }}
            >
              <svg
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M21 21l-4.35-4.35"
                  stroke="#6B7280"
                  strokeWidth="1.6"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <circle
                  cx="11"
                  cy="11"
                  r="5"
                  stroke="#6B7280"
                  strokeWidth="1.6"
                />
              </svg>
            </button>

            <button
              aria-label="Уведомления"
              className="w-10 h-10 rounded-full flex items-center justify-center hover:bg-gray-100"
            >
              <svg
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M15 17H9a3 3 0 01-3-3V10a6 6 0 1112 0v4a3 3 0 01-3 3z"
                  stroke="#111827"
                  strokeWidth="1.4"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <path
                  d="M13.73 21a2 2 0 01-3.46 0"
                  stroke="#111827"
                  strokeWidth="1.4"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>

            <div className="w-10 h-10 rounded-full bg-gray-200" />
          </div>
        </div>
      </div>

      {showSearch && (
        <div className="lg:hidden px-4 pb-3">
          <div className="relative">
            <span className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500 pointer-events-none">
              <svg
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M21 21l-4.35-4.35"
                  stroke="#6B7280"
                  strokeWidth="1.6"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <circle
                  cx="11"
                  cy="11"
                  r="5"
                  stroke="#6B7280"
                  strokeWidth="1.6"
                />
              </svg>
            </span>
            <input
              aria-label="Поиск"
              placeholder="Поиск"
              className="pl-10 pr-4 py-2 rounded-full bg-gray-100 text-sm w-full focus:outline-none"
            />
          </div>
        </div>
      )}

      <div
        ref={mobileRef}
        className={`lg:hidden transition-max-h duration-200 ease-in-out ${
          mobileOpen ? "max-h-60" : "max-h-0"
        } overflow-hidden px-4  z-50 bg-white shadow-lg`}
        style={{ WebkitOverflowScrolling: "touch" }}
      >
        <nav className="flex flex-col gap-3 text-base font-medium">
          <a href="#" className="block py-2 px-3 rounded-md hover:bg-gray-50">
            Контакты
          </a>
          <a href="#" className="block py-2 px-3 rounded-md hover:bg-gray-50">
            Информация
          </a>
          <a href="#" className="block py-2 px-3 rounded-md hover:bg-gray-50">
            Подразделения
          </a>
        </nav>
      </div>
    </header>
  );
};

export default Header;
