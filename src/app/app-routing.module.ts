import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { HomeComponent } from './home/home.component';
import { PredictComponent } from './predict/predict.component';
import { ResultComponent } from './result/result.component';

const routes: Routes = [
  {path:'predict',component:PredictComponent},
  {path:'result',component:ResultComponent},
  {path:'home',component:HomeComponent},
  { path: '',   redirectTo: '/home', pathMatch: 'full' }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
