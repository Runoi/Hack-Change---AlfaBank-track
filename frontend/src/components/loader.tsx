const Spinner = ({
  size = "medium",
}: {
  size?: "small" | "medium" | "large";
}) => {
  const sizeClasses = {
    small: "w-6 h-6 border-2",
    medium: "w-12 h-12 border-4",
    large: "w-16 h-16 border-4",
  };

  return (
    <div className="flex justify-center items-center">
      <div
        className={`
          ${sizeClasses[size]}
          border-[#ef3124]
          border-t-transparent
          rounded-full
          animate-spin
        `}
      ></div>
    </div>
  );
};

export default Spinner;
