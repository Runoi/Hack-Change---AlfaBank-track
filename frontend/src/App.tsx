import React, { useContext, useState } from "react";
import { observer } from "mobx-react-lite";
import { RootStoreContext } from "./store/RootStore";
import { prepareRadar, prepareWaterfall } from "./shared/lib/chartHelpers";
import Header from "./components/header";
import { WaterfallChart } from "./components/waterfall-chart";
import { RadarExplain } from "./components/spider-chart";
import FancyLoader from "./components/loader";
import { GrLinkPrevious } from "react-icons/gr";
import NotFound from "./components/notFound";
import CarouselNavigation from "./components/carousel-navigation";

const ClientDashboard: React.FC = observer(() => {
  const {
    ClientStore: { data, getClientAnalysis, error, loading },
  } = useContext(RootStoreContext);

  const [currentIndex, setCurrentIndex] = useState(0);

  const waterfallData = data ? prepareWaterfall(data) : [];
  const radarData = data ? prepareRadar(data) : [];

  const handleEnter = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key !== "Enter") return;
    const value = (e.target as HTMLInputElement).value.trim();
    if (!value) return;
    getClientAnalysis(value);
  };

  return (
    <>
      <Header />
      <div className="max-w-7xl min-h-screen m-auto  p-6">
        <div className="mb-7">
          <div className="flex gap-4">
            <button className="w-10 h-10 rounded-full bg-[#f2f3f5] shadow-sm hover:bg-[#ef3124] hover:text-white flex items-center justify-center transition-colors cursor-pointer">
              <GrLinkPrevious />
            </button>
            <h1 className="text-3xl font-bold text-black">
              Предсказатель доходов
            </h1>{" "}
          </div>
          <div className="bg-[#ef3124] rounded-l-full flex w-full h-1 mt-2"></div>
        </div>
        <div className="grid grid-cols-12 gap-4">
          <div className="col-span-12 lg:col-span-8">
            <div className="mb-6 flex justify-between gap-6 h-24">
              <div className="p-4 rounded-2xl shadow-xl">
                <label className="block text-lg font-semibold text-gray-600 mb-1">
                  Выбор пользователя
                </label>
                <input
                  type="text"
                  placeholder="Введите ID..."
                  className="w-40 py-2 text-lg rounded-xl bg-white outline-none text-gray-800
               placeholder-gray-400"
                  onKeyDown={handleEnter}
                />
              </div>

              <div className="p-4 rounded-2xl shadow-xl">
                <h2 className="text-lg font-semibold text-gray-600">Доход</h2>
                <div className="text-3xl font-bold text-gray-900">
                  {data ? Number(data.predicted_income).toFixed(2) + " ₽" : ""}
                </div>
              </div>
            </div>

            <div
              className="w-full rounded-2xl shadow-xl bg-[#f2f3f5] p-4 
           gap-6"
            >
              {loading ? (
                <FancyLoader />
              ) : error ? (
                <NotFound id={currentIndex} />
              ) : data ? (
                <>
                  <div className="p-4">
                    <RadarExplain data={radarData} />
                  </div>
                  <div className="p-4">
                    <WaterfallChart data={waterfallData} />
                  </div>
                </>
              ) : (
                ""
              )}
            </div>
          </div>
          <div className="col-span-12 lg:col-span-4">
            <div className="mb-6 justify-between gap-6">
              <div className="p-4 rounded-2xl shadow-xl w-full h-24">
                <label className="block text-lg font-semibold text-gray-600 mb-1">
                  Предложения
                </label>
                {data ? (
                  <CarouselNavigation
                    currentIndex={currentIndex}
                    totalItems={data.offers.length}
                    setCurrentIndex={setCurrentIndex}
                  />
                ) : (
                  ""
                )}
              </div>
            </div>
            <div
              className="w-full rounded-2xl shadow-xl bg-[#f2f3f5] p-4 
           gap-6"
            >
              {loading ? (
                <FancyLoader />
              ) : error ? (
                ""
              ) : data ? (
                <div className="mb-6">
                  <p className="text-gray-600 mb-3 text-xl font-bold">
                    {data.offers[currentIndex].title}
                  </p>
                  <p className="text-gray-600 mb-3 text-lg">
                    {data.offers[currentIndex].client_message}
                  </p>
                  <p className="mb-3 text-lg text-[#ef3124] font-bold">
                    {data.offers[currentIndex].internal_comment}
                  </p>
                </div>
              ) : (
                ""
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  );
});

export default ClientDashboard;
