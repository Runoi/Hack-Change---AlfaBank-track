import { LuClipboardX } from "react-icons/lu";

const NotFound = () => {
  return (
    <div className="flex items-center justify-center">
      <div>
        <LuClipboardX size={24} color={"#ef3124"} />
      </div>
      <p className="text-[#ef3124] mx-3 text-md">
        Клиент по данному идентификатору не найден
      </p>
    </div>
  );
};

export default NotFound;
