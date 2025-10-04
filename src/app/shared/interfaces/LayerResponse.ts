export interface LayerResponse {
  status: boolean;
  photos: Array<{
    date: string;
    url: string;
  }>;
}
