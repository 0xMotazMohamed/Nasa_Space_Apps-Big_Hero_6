export interface PointsResponse {
  dates:Array<string>;
  point: {
    values: Array<{
      date: string;
      no2: {
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
      hcho: {
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
  city: {
    polygon_arr: {
      type: string;
      coordinates: any;
    };
    name: string;
    values: Array<{
      date: string;
      no2: {
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
      hcho: {
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
