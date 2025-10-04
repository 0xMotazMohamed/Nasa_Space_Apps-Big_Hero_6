import { HttpInterceptorFn } from '@angular/common/http';
import { finalize } from 'rxjs';
import { inject } from '@angular/core';
import { LoaderService } from '../services/loader.service';

export const loaderInterceptor: HttpInterceptorFn = (request, next) => {
  const loaderService = inject(LoaderService);
  // Define excluded endpoints
  const excludedEndpoints = [
    'assets',
  ];

  // Check if current request should be excluded
  const shouldExclude = excludedEndpoints.some(endpoint =>
    request.url.includes(endpoint)
  );
  if (!shouldExclude) {
    loaderService.show();
  }

  return next(request).pipe(
    finalize(() => {
      if (!shouldExclude) {
        loaderService.hide();
      }
    })
  );
};
