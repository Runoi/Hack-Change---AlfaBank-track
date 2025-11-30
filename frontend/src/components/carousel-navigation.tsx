import React from "react";
import { GrPrevious, GrNext } from "react-icons/gr";

interface CarouselNavigationProps {
  currentIndex: number;
  totalItems: number;
  setCurrentIndex: React.Dispatch<React.SetStateAction<number>>;
}

const CarouselNavigation: React.FC<CarouselNavigationProps> = ({
  currentIndex,
  totalItems,
  setCurrentIndex,
}) => {
  const handlePrev = () => {
    setCurrentIndex((prevIndex) =>
      prevIndex === 0 ? totalItems - 1 : prevIndex - 1
    );
  };

  const handleNext = () => {
    setCurrentIndex((prevIndex) =>
      prevIndex === totalItems - 1 ? 0 : prevIndex + 1
    );
  };

  return (
    <div className="flex items-center gap-4">
      <span className="text-md font-medium text-gray-600">
        {currentIndex + 1}/{totalItems}
      </span>

      <div className="flex gap-2">
        <button
          onClick={handlePrev}
          className="w-10 h-10 rounded-full bg-white shadow-sm hover:bg-[#ef3124] hover:text-white flex items-center justify-center transition-colors cursor-pointer"
        >
          <GrPrevious />
        </button>
        <button
          onClick={handleNext}
          className="w-10 h-10 rounded-full bg-white shadow-sm hover:bg-[#ef3124] hover:text-white flex items-center justify-center transition-colors cursor-pointer"
        >
          <GrNext />
        </button>
      </div>
    </div>
  );
};

export default CarouselNavigation;
