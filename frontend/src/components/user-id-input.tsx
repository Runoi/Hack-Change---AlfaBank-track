import React, { useState, useRef } from "react";

const UserIdInput: React.FC = () => {
  const [value, setValue] = useState("");
  const inputRef = useRef<HTMLInputElement | null>(null);

  const submit = (v: string) => {
    alert(v.trim() ? `ID: ${v.trim()}` : "ID не введён");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      submit(value);
      inputRef.current?.blur();
    }
  };

  const handleBlur = () => {
    submit(value);
  };

  return (
    <div className="inline-block w-full">
      <div
        className="rounded-full bg-white shadow-[0_6px_18px_rgba(0,0,0,0.12)] px-4 py-2"
        style={{ minWidth: 220 }}
      >
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          onBlur={handleBlur}
          placeholder="Введите ID..."
          aria-label="Введите ID пользователя"
          className="w-full bg-transparent text-gray-700 placeholder-gray-400 outline-none text-sm"
        />
      </div>
    </div>
  );
};

export default UserIdInput;
