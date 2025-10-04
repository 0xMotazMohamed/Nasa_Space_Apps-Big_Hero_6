export interface PolygonResponse {
  dates:Array<string>;
  polygon: {
    values: Array<{
      date: string;
      no2: {
        value: number;
        AQI: {
          value: number;
          category: string;
        };
      };
      hcho: {
        value: number;
        AQI: {
          value: number;
          category: string;
        };
      };
      o3: {
        value: number;
        AQI: {
          value: number;
          category: string;
        };
      };
      AQI_General: {
        value: number;
        category: string;
      };
    }>;
  };
}
